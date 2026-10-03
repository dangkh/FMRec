"""RecommendationMemorySystem: LLM client (OpenAI-compatible API), usage accounting, tracing."""

import os
import json
from typing import List, Dict, Optional, Any
from collections import defaultdict
import time
import urllib.request
import urllib.error

from fmrec.common import AutoModelForCausalLM, AutoTokenizer, TRANSFORMERS_AVAILABLE, torch
from fmrec.records import BehaviorMemory, TraceRecorder, UserInteraction
from fmrec.variants.legacy_memory_system import LegacyMemorySystemMixin


class RecommendationMemorySystem(LegacyMemorySystemMixin):
    """A-Mem adapted for Amazon product recommendation with Memory Evolution"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_gemini_embeddings: bool = None,
                 chat_api_base: Optional[str] = None,
                 embedding_api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 chat_model_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None):
        _ = use_gemini_embeddings  # kept for backward compatibility

        self.llm_name = model_name
        self.embedding_model_name = embedding_model_name or os.getenv("embedding_model_name") or embedding_model
        self.chat_model_name = chat_model_name or os.getenv("chat_model_name") or model_name
        self.chat_api_base = (chat_api_base or os.getenv("chat_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.embedding_api_base = (embedding_api_base or os.getenv("embedding_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

        self.use_api_chat = bool(self.chat_api_base)
        # MEMCF graph retrieval does not require neural embeddings. Legacy
        # similarity/evolution code paths use deterministic hash vectors instead
        # of loading SentenceTransformer, so MEMCF runs never block on local
        # embedding model initialization.
        self.use_api_embedding = False

        self.tokenizer = None
        self.model = None
        self.embedding_model = None

        if not self.use_api_chat:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "Local chat model requires transformers+torch, or set chat_api_base/api_base env vars."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                dtype=torch.float16,
                device_map="auto"
            )


        self.behavior_memories: List[BehaviorMemory] = []
        self.user_interaction_history: List[UserInteraction] = []
        self.next_thought_id = 0
        self.trace_recorder: Optional[TraceRecorder] = None
        self.memory_diagnostics = defaultdict(float)
        self.llm_usage = defaultdict(float)
        self.llm_usage_by_type: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _trace(self, event_type: str, payload: Dict[str, Any]) -> None:
        if getattr(self, "trace_recorder", None) is not None:
            self.trace_recorder.log(event_type, payload)


    def _record_llm_usage(
        self,
        call_type: str,
        prompt: str,
        role_prompt: str,
        output: str,
        duration_seconds: float,
        usage: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is None:
            prompt_tokens = self._estimate_token_count(str(role_prompt) + "\n" + str(prompt))
        if completion_tokens is None:
            completion_tokens = self._estimate_token_count(output)
        if total_tokens is None:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

        call_type = str(call_type or "generic")
        metrics = {
            "calls": 1,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "seconds": float(duration_seconds),
            "errors": 0 if success else 1,
        }
        for key, value in metrics.items():
            self.llm_usage[key] += value
            self.llm_usage_by_type[call_type][key] += value

        self._trace("llm_call", {
            "call_type": call_type,
            "success": success,
            "error": error,
            "model": self.chat_model_name if self.use_api_chat else self.llm_name,
            "prompt_chars": len(str(prompt or "")),
            "completion_chars": len(str(output or "")),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "seconds": float(duration_seconds),
        })

    def get_llm_usage_summary(self) -> Dict[str, Any]:
        total_calls = int(self.llm_usage.get("calls", 0))
        total_seconds = float(self.llm_usage.get("seconds", 0.0))
        total_tokens = int(self.llm_usage.get("total_tokens", 0))
        by_type = {
            key: {
                "calls": int(vals.get("calls", 0)),
                "prompt_tokens": int(vals.get("prompt_tokens", 0)),
                "completion_tokens": int(vals.get("completion_tokens", 0)),
                "total_tokens": int(vals.get("total_tokens", 0)),
                "seconds": float(vals.get("seconds", 0.0)),
                "errors": int(vals.get("errors", 0)),
            }
            for key, vals in sorted(self.llm_usage_by_type.items())
        }
        return {
            "calls": total_calls,
            "prompt_tokens": int(self.llm_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(self.llm_usage.get("completion_tokens", 0)),
            "total_tokens": total_tokens,
            "seconds": total_seconds,
            "errors": int(self.llm_usage.get("errors", 0)),
            "avg_seconds_per_call": total_seconds / total_calls if total_calls else 0.0,
            "avg_tokens_per_call": total_tokens / total_calls if total_calls else 0.0,
            "by_call_type": by_type,
        }

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"API HTTP {e.code} at {url}: {body}")
            except Exception as e:
                last_error = e
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"API request failed after 3 attempts at {url}: {last_error}") from last_error

    def qwen_generate(
        self,
        prompt: str,
        role_prompt="You are a helpful AI assistant.",
        max_new_tokens=8000,
        json_schema: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
        call_type: str = "generic",
    ) -> str:
        start_time = time.time()
        if self.use_api_chat:
            endpoint = f"{self.chat_api_base}/chat/completions"
            temperature = float(os.getenv("MEMCF_TEMPERATURE", "0.0"))
            payload = {
                "model": self.chat_model_name,
                "messages": [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 1.0,
            }
            llm_seed = os.getenv("MEMCF_LLM_SEED", "").strip()
            if llm_seed:
                try:
                    payload["seed"] = int(llm_seed)
                except ValueError as exc:
                    raise ValueError("MEMCF_LLM_SEED must be an integer") from exc
            repetition_penalty = os.getenv("MEMCF_REPETITION_PENALTY", "1.05").strip()
            if repetition_penalty:
                try:
                    payload["repetition_penalty"] = float(repetition_penalty)
                except ValueError:
                    pass
            if json_schema is not None and os.getenv("MEMCF_USE_GUIDED_JSON", "0") == "1":
                # vLLM's OpenAI-compatible server accepts guided_json as an
                # extra request field. Keep this opt-in because older servers
                # may reject unknown structured-output parameters.
                payload["guided_json"] = json_schema
            elif json_mode and os.getenv("MEMCF_USE_RESPONSE_FORMAT_JSON", "0") == "1":
                # Some OpenAI-compatible Qwen endpoints support JSON mode.
                # The prompt/role must include the word JSON for those servers.
                payload["response_format"] = {"type": "json_object"}
            try:
                result = self._post_json(endpoint, payload)
                content = result["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    output = "".join(
                        chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
                    )
                else:
                    output = str(content)
                self._record_llm_usage(
                    call_type=call_type,
                    prompt=prompt,
                    role_prompt=role_prompt,
                    output=output,
                    duration_seconds=time.time() - start_time,
                    usage=result.get("usage"),
                    success=True,
                )
                return output
            except Exception as e:
                self._record_llm_usage(
                    call_type=call_type,
                    prompt=prompt,
                    role_prompt=role_prompt,
                    output="",
                    duration_seconds=time.time() - start_time,
                    usage=None,
                    success=False,
                    error=str(e),
                )
                raise

        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            prompt_tokens = int(inputs["input_ids"].shape[-1])
            gen_ids = outputs[0][prompt_tokens:]
            output = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            completion_tokens = int(len(gen_ids))
            self._record_llm_usage(
                call_type=call_type,
                prompt=prompt,
                role_prompt=role_prompt,
                output=output,
                duration_seconds=time.time() - start_time,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                success=True,
            )
            return output
        except Exception as e:
            self._record_llm_usage(
                call_type=call_type,
                prompt=prompt,
                role_prompt=role_prompt,
                output="",
                duration_seconds=time.time() - start_time,
                usage=None,
                success=False,
                error=str(e),
            )
            raise


    
    
    
    
    
    


    def record_memory_diagnostics(self, retrieved: int, kept: int, skipped: int) -> None:
        self.memory_diagnostics["eval_users"] += 1
        self.memory_diagnostics["retrieved_total"] += retrieved
        self.memory_diagnostics["kept_total"] += kept
        self.memory_diagnostics["skipped_total"] += skipped
        if kept > 0:
            self.memory_diagnostics["users_with_kept_memory"] += 1
    
