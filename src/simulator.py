import json
import re
import random
from typing import List, Dict, Any, Optional

from src.schema import PersonaProfile, QuestionSchema, load_schema, SurveySchema
from src.config import get_llm, has_api_key


class Simulator:
    def __init__(self, schema_path: str, priors: Optional[Dict[str, Dict[str, float]]] = None):
        self.schema = load_schema(schema_path)
        self.use_llm = has_api_key()
        self.priors = priors or {}
        if self.use_llm:
            self.llm = get_llm(temperature=0.3)

    def _build_priors_text(self) -> str:
        if not self.priors:
            return ""
        lines = []
        for qid, dist in self.priors.items():
            top = sorted(dist.items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join([f"{oid}={p:.0%}" for oid, p in top])
            lines.append(f"{qid}: {top_str}")
        return "\n".join(lines)

    def _build_questions_block(self) -> str:
        blocks = []
        for q in self.schema.questions:
            opts = "\n".join([f"  {o.id}: {o.text}" for o in q.options])
            blocks.append(f"{q.question_id} ({q.type}): {q.text}\nВарианты:\n{opts}")
        return "\n\n".join(blocks)

    def _dummy_answer_all(self) -> Dict[str, List[str]]:
        result = {}
        for q in self.schema.questions:
            num = random.randint(2, 5) if q.type == "multi-choice" else 1
            result[q.question_id] = random.sample([o.id for o in q.options], min(num, len(q.options)))
        return result

    def _parse_llm_response(self, text: str) -> Dict[str, List[str]]:
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
        valid_ids = {}
        for q in self.schema.questions:
            valid_ids[q.question_id] = {o.id for o in q.options}
        result = {}
        for qid, ids_set in valid_ids.items():
            raw = data.get(qid, [])
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, list):
                raw = []
            selected = [x for x in raw if x in ids_set]
            result[qid] = selected
        return result

    def survey_one(self, persona: PersonaProfile, scenario: str = "",
                   neighbors_summary: str = "") -> Dict[str, Any]:
        if not self.use_llm:
            ans = self._dummy_answer_all()
            ans["persona_id"] = persona.persona_id
            return ans

        priors_text = self._build_priors_text()
        questions_block = self._build_questions_block()

        scenario_part = ""
        if scenario:
            scenario_part = f"\n\nСЦЕНАРИЙ (учти при ответах):\n{scenario}"

        neighbors_part = ""
        if neighbors_summary:
            neighbors_part = f"\n\nМнения похожих людей:\n{neighbors_summary}"

        prompt = f"""Ты — участник социологического опроса. Отвечай строго от лица описанного человека.

ПРОФИЛЬ:
Пол: {persona.demographics.gender}
Возраст: {persona.demographics.age}
Район: {persona.demographics.region}
Доход: {persona.demographics.income}
Образование: {persona.demographics.education}
Архетип: {persona.archetype_id}
Признаки: tech_optimism={persona.behaviors.get('tech_optimism', 0.5):.2f}, privacy_concern={persona.behaviors.get('privacy_concern', 0.5):.2f}, internet_activity={persona.behaviors.get('internet_activity', 0.5):.2f}, trust_in_ai={persona.behaviors.get('trust_in_ai', 0.5):.2f}
Описание: {persona.narrative}{scenario_part}{neighbors_part}

СТАТИСТИКА реальных ответов:
{priors_text}

ВОПРОСЫ:
{questions_block}

Выбирай варианты реалистично, учитывая профиль и статистику.
Для multi-choice выбирай от 2 до 6 вариантов.
Ответь ТОЛЬКО json:
{{"q1": ["1.2", "1.5"], "q2": ["2.1"], "q3": ["3.5"], "q4": ["4.1"]}}"""

        try:
            resp = self.llm.invoke(prompt)
            parsed = self._parse_llm_response(resp.content)
            all_ok = all(len(parsed.get(q.question_id, [])) > 0 for q in self.schema.questions)
            if all_ok:
                parsed["persona_id"] = persona.persona_id
                return parsed
        except Exception:
            pass

        ans = self._dummy_answer_all()
        ans["persona_id"] = persona.persona_id
        return ans

    def run_survey(self, personas: List[PersonaProfile], scenario: str = "",
                   neighbors_graph: Optional[Dict[int, List[int]]] = None,
                   progress_callback=None) -> List[Dict[str, Any]]:
        results = []
        for i, p in enumerate(personas):
            neighbors_summary = ""
            if neighbors_graph and results and i in neighbors_graph:
                neighbors_summary = self._make_neighbors_summary(neighbors_graph[i], results)
            ans = self.survey_one(p, scenario=scenario, neighbors_summary=neighbors_summary)
            results.append(ans)
            if progress_callback:
                progress_callback(i + 1, len(personas))
        return results

    def _make_neighbors_summary(self, neighbor_indices: List[int], existing_results: List[Dict]) -> str:
        if not neighbor_indices or not existing_results:
            return ""
        counts = {}
        total = 0
        for idx in neighbor_indices:
            if idx < len(existing_results):
                total += 1
                for qid in ["q1", "q2", "q3", "q4"]:
                    if qid not in counts:
                        counts[qid] = {}
                    for oid in existing_results[idx].get(qid, []):
                        counts[qid][oid] = counts[qid].get(oid, 0) + 1
        if total == 0:
            return ""
        lines = []
        for qid in ["q1", "q2", "q3", "q4"]:
            if qid in counts:
                top = sorted(counts[qid].items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join([f"{oid}({c}/{total})" for oid, c in top])
                lines.append(f"{qid}: {top_str}")
        return "; ".join(lines)
