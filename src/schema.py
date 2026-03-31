import json
import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QuestionOption(BaseModel):
    id: str
    text: str


class QuestionSchema(BaseModel):
    question_id: str
    text: str
    type: str = Field(..., description="single-choice, multi-choice, etc.")
    options: List[QuestionOption]


class SurveySchema(BaseModel):
    questions: List[QuestionSchema]


class DemographicProfile(BaseModel):
    age: int
    gender: str
    region: str
    income: str
    education: str


class PersonaProfile(BaseModel):
    persona_id: str
    demographics: DemographicProfile
    archetype_id: str = Field(default="a0")
    behaviors: Dict[str, float] = Field(default_factory=dict)
    narrative: str = Field(default="")
    embedding: Optional[List[float]] = None


def load_schema(path: str) -> SurveySchema:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return SurveySchema(**data)
