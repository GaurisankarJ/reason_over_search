"""Search-R1 overlay registration.

Single import-side-effect module. The launch script does

    import training.src.registry  # populates DATASET_REGISTRY, PROCESSOR_REGISTRY, ENV_REGISTRY

before kicking off training. Idempotent — re-imports silently skip already-registered
entries (so test collection / interactive reloads don't blow up).

Why we monkey-patch DATASET_REGISTRY directly: NeMo-RL has `register_env` and
`register_processor` but no `register_dataset` (DATASET_REGISTRY is a hardcoded
dict in nemo_rl/data/datasets/response_datasets/__init__.py). Mutating it is the
documented overlay path.
"""
from __future__ import annotations

from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY
from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor
from nemo_rl.environments.utils import ENV_REGISTRY, register_env

from training.src.datasets.search_r1 import SearchR1Dataset
from training.src.processors.search_r1 import search_r1_processor

ENV_NAME = "search_r1"
DATASET_NAME = "search_r1"
PROCESSOR_NAME = "search_r1_processor"
ENV_ACTOR_FQN = "training.src.environments.search_r1_env.SearchR1Environment"


def register() -> None:
    if DATASET_NAME not in DATASET_REGISTRY:
        DATASET_REGISTRY[DATASET_NAME] = SearchR1Dataset
    if PROCESSOR_NAME not in PROCESSOR_REGISTRY:
        register_processor(PROCESSOR_NAME, search_r1_processor)
    if ENV_NAME not in ENV_REGISTRY:
        register_env(ENV_NAME, ENV_ACTOR_FQN)


register()
