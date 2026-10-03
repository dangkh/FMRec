"""Legacy RecommendationMemorySystem methods (v1 behaviour memory, embeddings, save/load)."""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import pickle
import hashlib

from fmrec.common import extract_json_object, item_category, make_jsonable
from fmrec.prompts import (
    add_candidate_aliases,
    build_compact_score_prompt,
    parse_score_entries_from_text,
    score_entries_to_ranking,
)
from fmrec.records import BehaviorMemory, UserInteraction


def interaction_to_trace(interaction: "UserInteraction") -> Dict[str, Any]:
    return make_jsonable(asdict(interaction))


def behavior_memory_to_trace(memory: "BehaviorMemory") -> Dict[str, Any]:
    data = make_jsonable(asdict(memory))
    data["embedding"] = None
    data["embedding_dim"] = int(len(memory.embedding)) if memory.embedding is not None else 0
    data["interaction_sequence"] = [interaction_to_trace(i) for i in memory.interaction_sequence]
    return data


class LegacyMemorySystemMixin:
    """RecommendationMemorySystem methods for: Legacy RecommendationMemorySystem methods (v1 behaviour memory, embeddings, save/load)."""

    def _estimate_token_count(self, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            try:
                return int(len(tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        # Conservative fallback used when running through an API without tokenizer.
        return max(1, int(len(text) / 4))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _create_embedding(self, text: str) -> np.ndarray:
        # No neural embedding dependency in MEMCF. This only supports legacy
        # memory-link/evolution code paths; main graph retrieval is symbolic.
        return self._simple_hash_embedding(str(text), dim=384).astype(np.float32)

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def add_interaction(self, 
                       item_id: str,
                       item_name: str,
                       item_category: str,
                       action_type: str = "purchase",
                       rating: Optional[float] = None,
                       metadata: Optional[Dict] = None) -> UserInteraction:
        interaction = UserInteraction(
            item_id=item_id,
            item_name=item_name,
            item_category=item_category,
            action_type=action_type,
            rating=rating,
            metadata=metadata or {}
        )
        
        self.user_interaction_history.append(interaction)
        
        return interaction

    def create_behavior_thought(self, 
                               interaction_window: List[UserInteraction],
                               k_neighbors: int = 10) -> BehaviorMemory:
        interaction_summary = []
        for interaction in interaction_window:
            summary = {
                "item": interaction.item_name,
                "category": interaction.item_category,
                "action": interaction.action_type
            }
            interaction_summary.append(summary)
        
        prompt = f"""Analyze this failed recommendation interaction.
        Input: {json.dumps(interaction_summary, indent=2)}

        Context:
        - The interaction contains a wrong choice and the preferred correct item.
        - Your job is to capture why the wrong choice happened and what correction rule should be applied next time.
        - This memory will be retrieved for future ranking. It must be specific enough to avoid being applied to unrelated users.

        Return ONLY a JSON object in this format:
        {{
        "behavior_explanation": "2-3 concise sentences explaining why the wrong choice was made versus the correct item",
        "pattern_description": "2-3 concise sentences describing a correction rule, including when it applies and when it should NOT apply",
        "applicable_when": ["specific title/type/category/attribute evidence required before using this memory"],
        "not_applicable_when": ["conditions where this memory should be ignored"],
        "wrong_item_type": "short concrete type/attribute of the wrong choice",
        "correct_item_type": "short concrete type/attribute of the preferred item",
        "evidence_terms_required": ["concrete evidence terms that must appear in future history/candidates before applying this memory"],
        "specificity_score": 0.0 to 1.0,
        "overgeneralization_risk": 0.0 to 1.0,
        "keywords": ["kw1", "kw2", ...] (5-8 concrete fail-interaction signals, no generic words)
        }}

        Requirements:
        - Ground every statement in the input interaction.
        - Emphasize contrast between wrong and correct choice.
        - Avoid generic shopping summaries like 'user preference', 'category mismatch', or 'prioritize relevant items' unless tied to concrete terms.
        - Do not claim the rule applies unless future candidates/history contain the applicable evidence."""

        try:
            # response = self.model.generate_content(prompt)
            response = self.qwen_generate(
                prompt=prompt,
                role_prompt='You are a behavioral memory modeling system.',
                call_type="memory_create",
            )
            # time.sleep(5)
            result = extract_json_object(response)
            self._trace("memory_create_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "interaction_window": [interaction_to_trace(i) for i in interaction_window],
            })
            
            behavior_explanation = result.get("behavior_explanation", "")
            pattern_description = result.get("pattern_description", "")
            applicable_when = result.get("applicable_when", [])
            not_applicable_when = result.get("not_applicable_when", [])
            wrong_item_type = result.get("wrong_item_type", "")
            correct_item_type = result.get("correct_item_type", "")
            evidence_terms_required = result.get("evidence_terms_required", [])
            try:
                specificity_score = float(result.get("specificity_score", 0.0))
            except Exception:
                specificity_score = 0.0
            try:
                overgeneralization_risk = float(result.get("overgeneralization_risk", 0.0))
            except Exception:
                overgeneralization_risk = 0.0
            if applicable_when:
                pattern_description += f" Applicable when: {json.dumps(applicable_when, ensure_ascii=False)}."
            if not_applicable_when:
                pattern_description += f" Do not apply when: {json.dumps(not_applicable_when, ensure_ascii=False)}."
            if evidence_terms_required:
                pattern_description += f" Evidence required: {json.dumps(evidence_terms_required, ensure_ascii=False)}."
            if wrong_item_type or correct_item_type:
                pattern_description += (
                    f" Wrong item type: {wrong_item_type}. "
                    f"Correct item type: {correct_item_type}."
                )
            keywords = result.get("keywords", [])
            for extra_kw in [wrong_item_type, correct_item_type] + list(evidence_terms_required or []):
                if extra_kw and extra_kw not in keywords:
                    keywords.append(extra_kw)
            # extracted_preferences = result.get("extracted_preferences", [])
            
        except Exception as e:
            print(f"Error in behavior analysis: {e}")
            self._trace("memory_create_error", {
                "error": str(e),
                "prompt": prompt,
                "interaction_window": [interaction_to_trace(i) for i in interaction_window],
            })
            behavior_explanation = f"A failed interaction occurred with {len(interaction_window)} compared items."
            pattern_description = "Correction rule is unclear; prefer signals from the preferred item over the wrong choice."
            # extracted_preferences = []
            keywords = [i.item_category for i in interaction_window[:3]]
            applicable_when = []
            not_applicable_when = []
            wrong_item_type = ""
            correct_item_type = ""
            evidence_terms_required = []
            specificity_score = 0.0
            overgeneralization_risk = 1.0
        
        combined_text = f"{behavior_explanation} {pattern_description} {' '.join(keywords)}"
        embedding = self._create_embedding(combined_text)
        
        behavior_memory = BehaviorMemory(
            thought_id=self.next_thought_id,
            interaction_sequence=interaction_window.copy(),
            behavior_explanation=behavior_explanation,
            pattern_description=pattern_description,
            # extracted_preferences=extracted_preferences,
            keywords=keywords,
            embedding=embedding,
            applicable_when=applicable_when if isinstance(applicable_when, list) else [str(applicable_when)],
            not_applicable_when=not_applicable_when if isinstance(not_applicable_when, list) else [str(not_applicable_when)],
            wrong_item_type=str(wrong_item_type or ""),
            correct_item_type=str(correct_item_type or ""),
            evidence_terms_required=evidence_terms_required if isinstance(evidence_terms_required, list) else [str(evidence_terms_required)],
            specificity_score=max(0.0, min(1.0, specificity_score)),
            overgeneralization_risk=max(0.0, min(1.0, overgeneralization_risk)),
        )
        
        self.next_thought_id += 1
        self._trace("memory_created", {
            "memory": behavior_memory_to_trace(behavior_memory),
        })
        return behavior_memory

    def link_behavior_memories(self, 
                               new_memory: BehaviorMemory,
                               k: int = 5, wo_link=False) -> List[int]:
        """Link new behavior memory with similar past patterns"""
        if len(self.behavior_memories) == 0:
            return []
        
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(new_memory.embedding, memory.embedding)
            similarities.append((memory.thought_id, sim, memory))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_k = similarities[:min(k, len(similarities))]
        
        if len(nearest_k) == 0:
            return []
        if wo_link:
            linked = [thought_id for thought_id, _, _ in nearest_k]
            self._trace("memory_link_decision", {
                "new_memory": behavior_memory_to_trace(new_memory),
                "wo_link": True,
                "linked_thought_ids": linked,
                "reasoning": "wo_link enabled; using nearest memories without LLM link filtering.",
            })
            return linked
        
        nearest_info = []
        for thought_id, sim, memory in nearest_k:
            nearest_info.append({
                "thought_id": thought_id,
                "behavior_explanation": memory.behavior_explanation,
                "pattern": memory.pattern_description,
                # "preferences": memory.extracted_preferences,
                "similarity": float(sim)
            })
        
        prompt = f"""Determine if the new fail-interaction memory should be linked to past fail memories.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Similar Past Patterns:
        {json.dumps(nearest_info, indent=2)}

        Link ONLY if:
        - They share a similar error/correction pattern (same mismatch type or same correction signal).
        - They imply a consistent fix strategy across users or interactions.
        - Their wrong-vs-correct contrast is semantically aligned.
        Do NOT link if they describe unrelated failure reasons.

        Return JSON:
        {{
        "should_link": true/false,
        "linked_thought_ids": [list of IDs],
        "reasoning": "1-2 sentences explaining shared fail/correction evidence"
        }}
        Keep reasoning concise and specific."""

        try:
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(
                prompt=prompt,
                role_prompt='You are a behavioral memory modeling system.',
                call_type="memory_link",
            )

            result = extract_json_object(response)
            self._trace("memory_link_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "new_memory": behavior_memory_to_trace(new_memory),
                "nearest_info": nearest_info,
            })
            
            if result.get("should_link", False):
                linked = result.get("linked_thought_ids", [])
            else:
                linked = []
            self._trace("memory_link_decision", {
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_thought_ids": linked,
                "reasoning": result.get("reasoning", ""),
            })
            return linked
                
        except Exception as e:
            print(f"Error in linking: {e}")
            linked = [thought_id for thought_id, sim, _ in nearest_k if sim > 0.65]
            self._trace("memory_link_error", {
                "error": str(e),
                "new_memory": behavior_memory_to_trace(new_memory),
                "nearest_info": nearest_info,
                "fallback_linked_thought_ids": linked,
            })
            return linked

    def evolve_behavior_memories(self,
                                new_memory: BehaviorMemory,
                                linked_ids: List[int],
                                max_evolutions_per_memory: Optional[int] = None) -> None:
        """Evolve existing behavior memories based on new patterns (Section 3.3)"""
        if len(linked_ids) == 0:
            return

        linked_memories = [m for m in self.behavior_memories if m.thought_id in linked_ids]
        if len(linked_memories) == 0:
            return
        # ============ LỌC MEMORIES CÒN CÓ THỂ EVOLVE ============
        evolvable_memories = []
        for mem in linked_memories:
            if max_evolutions_per_memory is not None:
                mem.max_evolutions = max_evolutions_per_memory
            
            if mem.can_evolve():
                evolvable_memories.append(mem)
            else:
                print(f"  ⚠ Memory {mem.thought_id} reached max evolutions ({mem.evolution_count}), skipping...")
        
        if len(evolvable_memories) == 0:
            print("  → No memories available for evolution (all reached max)")
            return
        
        
        mem_info = []
        for mem in evolvable_memories:
            mem_info.append({
                "thought_id": mem.thought_id,
                "behavior_explanation": mem.behavior_explanation,
                "pattern": mem.pattern_description,
                "evolution_count": mem.evolution_count 
            })
        

# Return ONLY JSON."""
        prompt = f"""Determine if past fail memories should be updated using a new fail case.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Linked Past Patterns (with evolution history):
        {json.dumps(mem_info, indent=2)}

        Update Guidelines:
        - Update when the new fail case provides clearer correction evidence for an existing fail pattern.
        - Refine wording toward a stronger wrong-vs-correct contrast.
        - Prefer updates that improve future error avoidance rules.
        - Skip updates when the new fail case is unrelated.

        Return JSON:
        {{
        "should_evolve": true/false,
        "updates": [
            {{
            "thought_id": ID,
            "behavior_explanation": "updated text or null",
            "new_pattern": "updated text or null",
            "reasoning": "1 sentence explaining how the fail-correction rule is refined"
            }}
        ]
        }}
        Ensure updates are grounded in input data and reasoning is concise."""

        try:
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(
                prompt=prompt,
                role_prompt='You are a behavioral memory modeling system.',
                call_type="memory_evolve",
            )
            result = extract_json_object(response)
            self._trace("memory_evolve_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_ids": linked_ids,
                "linked_memories": mem_info,
            })
            
            if result.get("should_evolve", False):
                updates = result.get("updates", [])
                
                for update in updates:
                    thought_id = update.get("thought_id")
                    memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
                    
                    if memory:
                        # ============ GHI LẠI GIÁ TRỊ CŨ ============
                        old_values = {
                            'behavior_explanation': memory.behavior_explanation,
                            'pattern_description': memory.pattern_description,
                            # 'extracted_preferences': memory.extracted_preferences.copy()
                        }
                    
                        updated = False
                        update_type = []
                        if update.get("behavior_explanation"):
                            memory.behavior_explanation = update["behavior_explanation"]
                            updated = True
                            update_type.append("behavior_explanation")

                        if update.get("new_pattern"):
                            memory.pattern_description = update["new_pattern"]
                            updated = True
                            update_type.append("pattern")
                        
                        # if update.get("additional_preferences"):
                        #     memory.extracted_preferences.extend(update["additional_preferences"])
                        #     memory.extracted_preferences = list(set(memory.extracted_preferences))
                        #     updated = True
                        #     update_type.append("preferences")
                        
                        # Regenerate embedding if updated
                        if updated:
                            # combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)} {' '.join(memory.extracted_preferences)}"
                            combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)}"
                            memory.embedding = self._create_embedding(combined_text)
                            new_values = {
                                'behavior_explanation': memory.behavior_explanation,
                                'pattern_description': memory.pattern_description,
                                # 'extracted_preferences': memory.extracted_preferences.copy()
                            }
                            
                            memory.record_evolution(
                                update_type=", ".join(update_type),
                                old_values=old_values,
                                new_values=new_values,
                                reasoning=update.get("reasoning", "")
                            )
                            self._trace("memory_evolved", {
                                "new_memory": behavior_memory_to_trace(new_memory),
                                "evolved_thought_id": thought_id,
                                "update_type": ", ".join(update_type),
                                "old_values": old_values,
                                "new_values": new_values,
                                "reasoning": update.get("reasoning", ""),
                                "evolution_count": memory.evolution_count,
                            })
                        
        except Exception as e:
            print(f"Error in memory evolution: {e}")
            self._trace("memory_evolve_error", {
                "error": str(e),
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_ids": linked_ids,
            })

    def add_behavior_memory(self,
                           interaction_window: List[UserInteraction],
                           k_neighbors: int = 5) -> BehaviorMemory:
        """Complete A-Mem pipeline: Create, Link, and Evolve"""
        # Step 1: Create behavior thought
        behavior_memory = self.create_behavior_thought(interaction_window, k_neighbors)
        
        # Step 2: Link with similar patterns
        linked_ids = self.link_behavior_memories(behavior_memory, k_neighbors)
        behavior_memory.links = linked_ids
        
        # Update bidirectional links
        for thought_id in linked_ids:
            memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
            if memory and behavior_memory.thought_id not in memory.links:
                memory.links.append(behavior_memory.thought_id)
        
        # Step 3: Evolve existing memories based on new pattern
        self.evolve_behavior_memories(behavior_memory, linked_ids)
        
        # Add to collection
        self.behavior_memories.append(behavior_memory)
        return behavior_memory

    def retrieve_relevant_memory_records(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memory records with similarity scores."""
        if len(self.behavior_memories) == 0:
            return []
        
        profile_embedding = self._create_embedding(query_text)
        
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(profile_embedding, memory.embedding)
            similarities.append((memory, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_records = [
            {"memory": mem, "similarity": float(sim)}
            for mem, sim in similarities[:k]
        ]
        self._trace("memory_retrieval", {
            "query_text": query_text,
            "k": k,
            "retrieved": [
                {
                    "similarity": float(sim),
                    "memory": behavior_memory_to_trace(mem),
                }
                for mem, sim in similarities[:k]
            ],
        })
        return top_records

    def retrieve_relevant_memories(self, user_profile_text: str, k: int = 5) -> List[BehaviorMemory]:
        """Backward-compatible retrieval API returning only memories."""
        return [record["memory"] for record in self.retrieve_relevant_memory_records(user_profile_text, k=k)]

    def llm_ranking(self,
                   train_items: List[Dict],
                   candidate_items: List[Dict],
                   retrieved_memories: Optional[List[BehaviorMemory]],
                   prompt_sample: str,
                   ranking_prompt_style: str = "memcf",
                   trace_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Score candidates with the LLM, then sort locally.

        Previous versions asked Qwen to output a full permutation of raw item IDs.
        Traces showed frequent duplicates/missing IDs. This score-based path asks
        for C01..C20 candidate scores and maps them back to item IDs in code.
        """
        user_profile = [
            {"title": item["title"], "category": item["category"]}
            for item in train_items
        ]
        candidate_info = [
            {"item_id": item["item_id"], "title": item["title"], "category": item["category"]}
            for item in candidate_items
        ]
        aliased_candidates, alias_to_item_id = add_candidate_aliases(candidate_info)
        valid_candidate_aliases = list(alias_to_item_id.keys())

        memory_thoughts = []
        if retrieved_memories:
            for mem in retrieved_memories:
                memory_thoughts.append({
                    "memory_id": mem.thought_id,
                    "behavior_explanation": mem.behavior_explanation,
                    "pattern": mem.pattern_description,
                    "applicable_when": getattr(mem, "applicable_when", []),
                    "not_applicable_when": getattr(mem, "not_applicable_when", []),
                    "wrong_item_type": getattr(mem, "wrong_item_type", ""),
                    "correct_item_type": getattr(mem, "correct_item_type", ""),
                    "evidence_terms_required": getattr(mem, "evidence_terms_required", []),
                    "specificity_score": getattr(mem, "specificity_score", 0.0),
                    "overgeneralization_risk": getattr(mem, "overgeneralization_risk", 0.0),
                    "keywords": getattr(mem, "keywords", [])[:10],
                })

        if ranking_prompt_style == "compact_score":
            prompt = build_compact_score_prompt(
                history_items=user_profile[-10:],
                aliased_candidates=aliased_candidates,
                prompt_sample=prompt_sample,
                memory_payload=memory_thoughts if retrieved_memories else None,
                user_profile_payload=None,
            )
        elif retrieved_memories:
            prompt = f"""
You are scoring candidate items for a recommender system.

Important memory policy:
- Retrieved memories may be irrelevant.
- Use a memory only if its applicable_when or evidence_terms_required directly appears in the current user history or candidate items.
- If a memory conflicts with recent user history or candidate facts, ignore the memory.
- If no memory is clearly applicable, score exactly as you would from user history and candidate facts only.
- Memories are weak evidence, not hard rules.

Inputs:
User Recent History (last interactions; prioritize most recent):
{json.dumps(user_profile[-10:], ensure_ascii=False, indent=2)}

Retrieved Fail-Correction Memories:
{json.dumps(memory_thoughts, ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""
        else:
            prompt = f"""
You are scoring candidate items for a recommender system based only on user history and candidate facts.
{prompt_sample}

Inputs:
User Recent History (last interactions; prioritize most recent):
{json.dumps(user_profile[-10:], ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""

        score_json_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string", "enum": valid_candidate_aliases},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "rationale": {"type": "string"},
                        },
                        "required": ["candidate_id", "score", "rationale"],
                        "additionalProperties": False,
                    },
                    "minItems": len(valid_candidate_aliases),
                    "maxItems": len(valid_candidate_aliases),
                },
            },
            "required": ["scores"],
            "additionalProperties": False,
        }
        if ranking_prompt_style != "compact_score":
            score_json_schema["properties"]["reasoning"] = {"type": "string"}
            score_json_schema["required"] = ["scores", "reasoning"]

        max_retries = int(os.getenv("MEMCF_RANK_RETRIES", "1"))
        current_prompt = prompt
        last_error: Optional[str] = None
        final_ranked: Optional[List[str]] = None
        final_validation: Optional[Dict[str, Any]] = None
        final_result: Dict[str, Any] = {}
        raw_response = ""

        for attempt in range(max_retries + 1):
            try:
                raw_response = self.qwen_generate(
                    prompt=current_prompt,
                    role_prompt=(
                        "You are a deterministic recommender scorer. "
                        "Return JSON only and follow the provided JSON schema exactly."
                    ),
                    max_new_tokens=int(os.getenv("MEMCF_RANK_MAX_TOKENS", "1400")),
                    json_schema=score_json_schema,
                    json_mode=True,
                    call_type="ranking",
                )
                try:
                    result = extract_json_object(raw_response)
                    raw_scores = result.get("scores", [])
                except Exception as parse_error:
                    last_error = str(parse_error)
                    result = {
                        "scores": parse_score_entries_from_text(raw_response, alias_to_item_id),
                        "reasoning": "Recovered score rows from malformed JSON",
                    }
                    raw_scores = result.get("scores", [])

                ranked_ids, validation = score_entries_to_ranking(raw_scores, alias_to_item_id)
                final_ranked = ranked_ids
                final_validation = validation
                final_result = result

                self.memory_diagnostics["rank_score_calls"] += 1
                self.memory_diagnostics["rank_missing_score_rows"] += len(validation["missing_candidate_ids"])
                self.memory_diagnostics["rank_invalid_score_rows"] += len(validation["invalid_or_duplicate_rows"])
                if validation["is_valid"]:
                    self.memory_diagnostics["rank_valid_score_outputs"] += 1
                else:
                    self.memory_diagnostics["rank_invalid_score_outputs"] += 1

                self._trace("ranking_llm", {
                    **(trace_context or {}),
                    "attempt": attempt,
                    "ranking_mode": "score_based_candidate_alias",
                    "prompt": current_prompt,
                    "role_prompt": (
                        "You are a deterministic recommender scorer. "
                        "Return JSON only and follow the provided JSON schema exactly."
                    ),
                    "answer": raw_response,
                    "parsed": result,
                    "score_validation": validation,
                    "raw_output_valid": validation["is_valid"],
                    "cleaned_ranked_item_ids": ranked_ids,
                    "candidate_items": candidate_info,
                    "aliased_candidate_items": aliased_candidates,
                    "alias_to_item_id": alias_to_item_id,
                    "train_items": user_profile[-10:],
                    "retrieved_memories": [
                        behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
                    ],
                    "use_retrieved_memories": retrieved_memories is not None,
                })

                if validation["is_valid"] or attempt >= max_retries:
                    if not validation["is_valid"]:
                        self._trace("ranking_retry_exhausted", {
                            **(trace_context or {}),
                            "attempts": attempt + 1,
                            "final_validation": validation,
                            "cleaned_ranked_item_ids": ranked_ids,
                        })
                    return ranked_ids

                current_prompt = f"""{prompt}

The previous answer was invalid:
{json.dumps(validation, ensure_ascii=False, indent=2)}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""
            except Exception as e:
                last_error = str(e)
                self.memory_diagnostics["rank_attempt_errors"] += 1
                self._trace("ranking_attempt_error", {
                    **(trace_context or {}),
                    "attempt": attempt,
                    "error": last_error,
                    "prompt": current_prompt,
                    "candidate_items": candidate_info,
                    "aliased_candidate_items": aliased_candidates,
                    "retrieved_memories": [
                        behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
                    ],
                })
                if attempt >= max_retries:
                    break
                current_prompt = f"""{prompt}

The previous answer could not be parsed because:
{last_error}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""

        print(f"Error in LLM scoring/ranking: {last_error}")
        fallback_ids = [str(item["item_id"]) for item in candidate_items]
        self.memory_diagnostics["rank_fallbacks"] += 1
        self._trace("ranking_error", {
            **(trace_context or {}),
            "error": last_error,
            "prompt": prompt,
            "answer": raw_response,
            "parsed": final_result,
            "score_validation": final_validation,
            "candidate_items": candidate_info,
            "aliased_candidate_items": aliased_candidates,
            "fallback_ranked_item_ids": fallback_ids,
            "retrieved_memories": [
                behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
            ],
        })
        return fallback_ids

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Phân tích thống kê về evolution của các memories"""
        if not self.behavior_memories:
            return {}
        
        evolution_counts = [m.evolution_count for m in self.behavior_memories]
        
        stats = {
            'total_memories': len(self.behavior_memories),
            'total_evolutions': sum(evolution_counts),
            'avg_evolutions_per_memory': np.mean(evolution_counts),
            'max_evolutions': max(evolution_counts),
            'min_evolutions': min(evolution_counts),
            'std_evolutions': np.std(evolution_counts),
            'memories_never_evolved': sum(1 for c in evolution_counts if c == 0),
            'memories_evolved_once': sum(1 for c in evolution_counts if c == 1),
            'memories_evolved_multiple': sum(1 for c in evolution_counts if c > 1),
            'evolution_distribution': {
                f'{i}_times': sum(1 for c in evolution_counts if c == i)
                for i in range(max(evolution_counts) + 1)
            }
        }
        
        # Top memories theo evolution count
        top_evolved = sorted(
            [(m.thought_id, m.evolution_count, m.behavior_explanation) 
            for m in self.behavior_memories],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats['top_10_most_evolved'] = [
            {
                'thought_id': tid,
                'evolution_count': count,
                'behavior': behavior[:100]  # Truncate
            }
            for tid, count, behavior in top_evolved
        ]
        
        return stats

    def print_evolution_report(self):
        """In báo cáo evolution"""
        stats = self.get_evolution_statistics()
        
        print("\n" + "="*80)
        print("MEMORY EVOLUTION REPORT")
        print("="*80)
        print(f"Total Memories: {stats['total_memories']}")
        print(f"Total Evolutions: {stats['total_evolutions']}")
        print(f"Average Evolutions per Memory: {stats['avg_evolutions_per_memory']:.2f}")
        print(f"Max Evolutions: {stats['max_evolutions']}")
        print(f"Min Evolutions: {stats['min_evolutions']}")
        print(f"Std Deviation: {stats['std_evolutions']:.2f}")
        print("-"*80)
        print(f"Never Evolved: {stats['memories_never_evolved']}")
        print(f"Evolved Once: {stats['memories_evolved_once']}")
        print(f"Evolved Multiple Times: {stats['memories_evolved_multiple']}")
        print("-"*80)
        print("Evolution Distribution:")
        for times, count in stats['evolution_distribution'].items():
            if count > 0:
                print(f"  {times}: {count} memories")
        print("-"*80)
        print("Top 10 Most Evolved Memories:")
        for item in stats['top_10_most_evolved']:
            print(f"  ID {item['thought_id']}: {item['evolution_count']} evolutions")
            print(f"    → {item['behavior']}")

    def save_memory(self, filepath: str, format: str = 'json') -> None:
        """
        Lưu memory system ra file (chứa memories của TẤT CẢ users)
        
        Args:
            filepath: Đường dẫn file để lưu
            format: Định dạng file ('json' hoặc 'pickle')
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if format == 'json':
            memories_dict = [mem.to_dict() for mem in self.behavior_memories]
            interactions_dict = [asdict(interaction) for interaction in self.user_interaction_history]
            
            data = {
                'behavior_memories': memories_dict,
                'user_interaction_history': interactions_dict,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
            
        elif format == 'pickle':
            data = {
                'behavior_memories': self.behavior_memories,
                'user_interaction_history': self.user_interaction_history,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

    def load_memory(self, filepath: str, format: str = None) -> None:
        """
        Tải memory system từ file
        
        Args:
            filepath: Đường dẫn file để đọc
            format: Định dạng file ('json' hoặc 'pickle'). Nếu None, tự động detect từ extension
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format is None:
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            else:
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                    format = 'json'
                except:
                    format = 'pickle'
        
        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.behavior_memories = [
                BehaviorMemory.from_dict(mem_dict) 
                for mem_dict in data['behavior_memories']
            ]
            
            self.user_interaction_history = [
                UserInteraction(**interaction_dict)
                for interaction_dict in data['user_interaction_history']
            ]
            
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.behavior_memories = data['behavior_memories']
            self.user_interaction_history = data['user_interaction_history']
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
