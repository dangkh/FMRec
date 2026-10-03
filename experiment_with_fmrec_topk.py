"""FMRec experiment entry point.

The implementation lives in the ``fmrec`` package next to this file:
``fmrec/`` holds the pool-LLM path (FMRec main method) and ``fmrec/variants/``
holds every experimental variant. Command-line usage is unchanged.
"""
from fmrec.cli import main_v2


if __name__ == "__main__":
    main_v2()
