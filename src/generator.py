import os
import uuid
import random
import pandas as pd
from typing import List, Optional, Callable

from src.schema import DemographicProfile, PersonaProfile
from src.archetypes import generate_archetypes, Archetype
from src.behaviors import sample_behaviors
from src.embeddings import Embedder


def pick_archetype(archetypes: List[Archetype], demo: DemographicProfile, rng: random.Random) -> Archetype:
    a = {x.archetype_id: x for x in archetypes}
    age = demo.age
    edu = (demo.education or "").lower()
    inc = (demo.income or "").lower()

    p_lowtech = 0.08 + 0.012 * max(0, age - 50)
    p_skeptic = 0.05 + 0.008 * max(0, age - 55)
    p_optimist = 0.20
    p_social = 0.15

    if "высш" in edu:
        p_optimist += 0.10
        p_social += 0.05
    if "высок" in inc or "достаточн" in inc:
        p_optimist += 0.08

    if age < 30:
        p_social += 0.15
        p_lowtech = 0.02

    r = rng.random()
    cumul = 0.0
    cumul += min(0.35, p_lowtech)
    if r < cumul:
        return a.get("a5", archetypes[4 % len(archetypes)])
    cumul += min(0.15, p_skeptic)
    if r < cumul:
        return a.get("a8", archetypes[7 % len(archetypes)])
    cumul += min(0.30, p_optimist)
    if r < cumul:
        return a.get("a1", archetypes[0])
    cumul += min(0.25, p_social)
    if r < cumul:
        return a.get("a3", archetypes[2 % len(archetypes)])

    return rng.choice(archetypes)


def sample_demographics(df: pd.DataFrame, n: int) -> List[DemographicProfile]:
    sampled = df.sample(n=n, replace=True).reset_index(drop=True)
    profiles = []
    for _, row in sampled.iterrows():
        profiles.append(DemographicProfile(
            age=int(row["Возраст"]),
            gender=str(row["Пол"]),
            region=str(row["Район проживания"]),
            income=str(row["Вопрос 5"]),
            education=str(row["Вопрос 6"]),
        ))
    return profiles


def build_narrative(archetype: Archetype, demo: DemographicProfile, behaviors: dict, rng: random.Random) -> str:
    """Строит РУССКИЙ нарратив из архетипа + демографии + поведения."""
    phrase = rng.choice(archetype.signature_phrases) if archetype.signature_phrases else ""
    parts = []

    parts.append(f"{demo.gender}, {demo.age} лет, район {demo.region}.")
    parts.append(f"Доход: {demo.income}. Образование: {demo.education}.")
    parts.append(f"Архетип: {archetype.name} — {archetype.description}.")

    if phrase:
        parts.append(f'Характерная фраза: «{phrase}».')

    extras = []
    if behaviors.get("privacy_concern", 0.5) > 0.65:
        extras.append("с осторожностью относится к сбору персональных данных")
    if behaviors.get("tech_optimism", 0.5) > 0.7:
        extras.append("в целом позитивно смотрит на цифровизацию")
    if behaviors.get("internet_activity", 0.5) < 0.35:
        extras.append("предпочитает решать часть вопросов офлайн")
    elif behaviors.get("internet_activity", 0.5) > 0.75:
        extras.append("проводит много времени в интернете")
    if behaviors.get("trust_in_ai", 0.5) < 0.30:
        extras.append("решения ИИ скорее перепроверит")
    elif behaviors.get("trust_in_ai", 0.5) > 0.65:
        extras.append("доверяет рекомендациям ИИ")

    if extras:
        parts.append(" ".join(e.capitalize() + "." for e in extras))

    return " ".join(parts)


def generate_population_archetypes(
    df: pd.DataFrame,
    n: int = 1000,
    k_archetypes: int = 20,
    output_file: str = "data/synthetic_population.jsonl",
    seed: int = 42,
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    progress_callback: Optional[Callable[[int, int, PersonaProfile], None]] = None,
) -> List[PersonaProfile]:
    os.makedirs(os.path.dirname(output_file) or "data", exist_ok=True)

    rng = random.Random(seed)
    archetypes = generate_archetypes(k=k_archetypes, output_dir=(os.path.dirname(output_file) or "data"))
    demos = sample_demographics(df, n=n)

    personas: List[PersonaProfile] = []
    narratives: List[str] = []

    for i, demo in enumerate(demos):
        arch = pick_archetype(archetypes, demo, rng)
        behaviors = sample_behaviors(arch, rng=rng)
        narrative = build_narrative(arch, demo, behaviors, rng)

        p = PersonaProfile(
            persona_id=str(uuid.uuid4()),
            demographics=demo,
            archetype_id=arch.archetype_id,
            behaviors=behaviors,
            narrative=narrative,
            embedding=None,
        )
        personas.append(p)
        narratives.append(narrative)

        if progress_callback:
            progress_callback(i + 1, n, p)

    # embeddings батчом
    embedder = Embedder(model_name=embed_model)
    vecs = embedder.embed_texts(narratives)
    for p, v in zip(personas, vecs):
        p.embedding = v

    with open(output_file, "w", encoding="utf-8") as f:
        for p in personas:
            f.write(p.model_dump_json() + "\n")

    return personas
