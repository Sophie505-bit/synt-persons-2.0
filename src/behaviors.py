import random
from typing import Dict
from src.archetypes import Archetype


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sample_behaviors(archetype: Archetype, noise: float = 0.12, rng: random.Random = None) -> Dict[str, float]:
    rng = rng or random.Random()
    b = {}
    for k, v in archetype.traits.items():
        b[k] = clamp01(v + rng.uniform(-noise, noise))
    return b
