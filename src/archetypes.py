import json
import os
import random
import time
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm, has_api_key


class Archetype(BaseModel):
    archetype_id: str
    name: str
    description: str
    traits: Dict[str, float] = Field(default_factory=dict)
    signature_phrases: List[str] = Field(default_factory=list)


# ---- РУССКИЕ fallback-архетипы (20 штук) ----
FALLBACK_ARCHETYPES: List[Archetype] = [
    Archetype(archetype_id="a1", name="цифровой энтузиаст",
              description="активно использует все цифровые сервисы, первым пробует новинки, верит что технологии улучшают жизнь",
              traits={"tech_optimism": 0.92, "privacy_concern": 0.25, "internet_activity": 0.90, "trust_in_ai": 0.82},
              signature_phrases=["технологии — это будущее", "чем больше цифры, тем лучше"]),
    Archetype(archetype_id="a2", name="защитник приватности",
              description="ценит личные данные превыше удобства, скептичен к сбору информации, пользуется VPN",
              traits={"tech_optimism": 0.40, "privacy_concern": 0.92, "internet_activity": 0.55, "trust_in_ai": 0.20},
              signature_phrases=["мои данные — моё дело", "не доверяю тем, кто собирает информацию"]),
    Archetype(archetype_id="a3", name="соцсетевой активист",
              description="живёт в соцсетях и мессенджерах, постоянно на связи, делится контентом",
              traits={"tech_optimism": 0.72, "privacy_concern": 0.38, "internet_activity": 0.95, "trust_in_ai": 0.55},
              signature_phrases=["я всегда онлайн", "новости узнаю из телеграма"]),
    Archetype(archetype_id="a4", name="цифровой минималист",
              description="сознательно ограничивает время онлайн, использует только необходимые сервисы",
              traits={"tech_optimism": 0.35, "privacy_concern": 0.75, "internet_activity": 0.30, "trust_in_ai": 0.30},
              signature_phrases=["меньше экранов — больше жизни", "цифровой детокс необходим"]),
    Archetype(archetype_id="a5", name="low-tech пользователь",
              description="слабо разбирается в технологиях, нужна помощь, пользуется только базовыми функциями",
              traits={"tech_optimism": 0.30, "privacy_concern": 0.50, "internet_activity": 0.18, "trust_in_ai": 0.22},
              signature_phrases=["мне сложно разобраться", "прошу внука помочь настроить"]),
    Archetype(archetype_id="a6", name="онлайн-покупатель",
              description="активно покупает онлайн, ценит удобство доставки и маркетплейсов",
              traits={"tech_optimism": 0.78, "privacy_concern": 0.45, "internet_activity": 0.72, "trust_in_ai": 0.62},
              signature_phrases=["зачем идти в магазин, если можно заказать", "маркетплейсы — это удобно"]),
    Archetype(archetype_id="a7", name="осторожный прагматик",
              description="пользуется цифровыми сервисами по необходимости, взвешивает риски и пользу",
              traits={"tech_optimism": 0.55, "privacy_concern": 0.68, "internet_activity": 0.52, "trust_in_ai": 0.42},
              signature_phrases=["если это удобно и безопасно — почему бы и нет", "сначала почитаю отзывы"]),
    Archetype(archetype_id="a8", name="цифровой скептик",
              description="не доверяет цифровизации, считает что технологии создают больше проблем чем решают",
              traits={"tech_optimism": 0.18, "privacy_concern": 0.85, "internet_activity": 0.28, "trust_in_ai": 0.15},
              signature_phrases=["не всё нужно переводить в цифру", "раньше было лучше без этого"]),
    Archetype(archetype_id="a9", name="удалённый работник",
              description="работает из дома, зависит от цифровых инструментов для работы и общения",
              traits={"tech_optimism": 0.75, "privacy_concern": 0.55, "internet_activity": 0.82, "trust_in_ai": 0.60},
              signature_phrases=["дом — мой офис", "без интернета работа встанет"]),
    Archetype(archetype_id="a10", name="геймер",
              description="много времени проводит в играх и игровых сообществах, технически подкован",
              traits={"tech_optimism": 0.80, "privacy_concern": 0.35, "internet_activity": 0.88, "trust_in_ai": 0.52},
              signature_phrases=["игры — мой способ отдыха", "в онлайне у меня свои сообщества"]),
    Archetype(archetype_id="a11", name="забота о здоровье",
              description="использует приложения и гаджеты для отслеживания здоровья и фитнеса",
              traits={"tech_optimism": 0.72, "privacy_concern": 0.58, "internet_activity": 0.60, "trust_in_ai": 0.58},
              signature_phrases=["фитнес-браслет всегда со мной", "технологии помогают следить за здоровьем"]),
    Archetype(archetype_id="a12", name="умный дом",
              description="внедряет умные устройства дома, автоматизирует быт",
              traits={"tech_optimism": 0.88, "privacy_concern": 0.42, "internet_activity": 0.68, "trust_in_ai": 0.72},
              signature_phrases=["управляю всем со смартфона", "умный дом экономит время"]),
    Archetype(archetype_id="a13", name="онлайн-ученик",
              description="активно учится онлайн, проходит курсы, смотрит образовательный контент",
              traits={"tech_optimism": 0.82, "privacy_concern": 0.38, "internet_activity": 0.78, "trust_in_ai": 0.65},
              signature_phrases=["учиться можно где угодно", "онлайн-курсы — это доступное образование"]),
    Archetype(archetype_id="a14", name="новостной маньяк",
              description="постоянно читает новости из разных источников онлайн",
              traits={"tech_optimism": 0.58, "privacy_concern": 0.52, "internet_activity": 0.75, "trust_in_ai": 0.45},
              signature_phrases=["нужно быть в курсе событий", "читаю новости из нескольких источников"]),
    Archetype(archetype_id="a15", name="создатель контента",
              description="создаёт и публикует свой контент: видео, тексты, подкасты",
              traits={"tech_optimism": 0.85, "privacy_concern": 0.40, "internet_activity": 0.92, "trust_in_ai": 0.55},
              signature_phrases=["я делюсь своими идеями с миром", "контент — это мой способ самовыражения"]),
    Archetype(archetype_id="a16", name="городской исследователь",
              description="использует технологии для навигации, поиска мест и городских сервисов",
              traits={"tech_optimism": 0.68, "privacy_concern": 0.42, "internet_activity": 0.65, "trust_in_ai": 0.50},
              signature_phrases=["навигатор — мой лучший друг", "люблю находить новые места через приложения"]),
    Archetype(archetype_id="a17", name="эко-сознательный",
              description="использует цифровые платформы для продвижения экологических инициатив",
              traits={"tech_optimism": 0.62, "privacy_concern": 0.50, "internet_activity": 0.58, "trust_in_ai": 0.55},
              signature_phrases=["технологии могут помочь планете", "через интернет продвигаю экологичный образ жизни"]),
    Archetype(archetype_id="a18", name="госуслуги-пользователь",
              description="активно пользуется электронными госуслугами, оплатой ЖКХ, записью к врачу онлайн",
              traits={"tech_optimism": 0.65, "privacy_concern": 0.60, "internet_activity": 0.62, "trust_in_ai": 0.48},
              signature_phrases=["госуслуги онлайн — очень удобно", "всё можно сделать не выходя из дома"]),
    Archetype(archetype_id="a19", name="тревожный пользователь",
              description="пользуется технологиями но постоянно тревожится о безопасности и влиянии на психику",
              traits={"tech_optimism": 0.42, "privacy_concern": 0.78, "internet_activity": 0.55, "trust_in_ai": 0.30},
              signature_phrases=["переживаю за свои данные", "негативные новости портят настроение"]),
    Archetype(archetype_id="a20", name="общественник-организатор",
              description="использует цифровые инструменты для организации сообществ и мероприятий",
              traits={"tech_optimism": 0.75, "privacy_concern": 0.40, "internet_activity": 0.72, "trust_in_ai": 0.50},
              signature_phrases=["вместе мы можем больше", "организую мероприятия через соцсети"]),
]


def _ensure_traits(a: dict, rng: random.Random) -> dict:
    t = a.get("traits") or {}
    def pick(key, default):
        v = t.get(key, None)
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return default
    a["traits"] = {
        "tech_optimism": pick("tech_optimism", rng.uniform(0.2, 0.8)),
        "privacy_concern": pick("privacy_concern", rng.uniform(0.2, 0.8)),
        "internet_activity": pick("internet_activity", rng.uniform(0.2, 0.8)),
        "trust_in_ai": pick("trust_in_ai", rng.uniform(0.2, 0.8)),
    }
    if not isinstance(a.get("signature_phrases"), list):
        a["signature_phrases"] = []
    return a


def _cache_path(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "archetypes_cache.json")


def load_cached_archetypes(output_dir: str = "data") -> "List[Archetype] | None":
    path = _cache_path(output_dir)
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            return [Archetype(**x) for x in data]
        except Exception:
            return None
    return None


def save_cached_archetypes(archetypes: "List[Archetype]", output_dir: str = "data") -> None:
    path = _cache_path(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([a.model_dump() for a in archetypes], f, ensure_ascii=False, indent=2)


def generate_archetypes(k: int = 20, seed: int = 42, output_dir: str = "data") -> "List[Archetype]":
    cached = load_cached_archetypes(output_dir=output_dir)
    if cached and len(cached) >= k:
        return cached[:k]

    if not has_api_key():
        save_cached_archetypes(FALLBACK_ARCHETYPES[:k], output_dir=output_dir)
        return FALLBACK_ARCHETYPES[:k]

    rng = random.Random(seed)
    llm = get_llm(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "ты социолог. сгенерируй архетипы цифрового поведения российских горожан. "
         "ВСЕ тексты строго на РУССКОМ языке. "
         "верни строго json (и ничего кроме json): список из k объектов. "
         "поля: archetype_id (str), name (str, русский), description (str, русский), "
         "traits (объект с ключами: tech_optimism, privacy_concern, internet_activity, trust_in_ai — числа 0..1), "
         "signature_phrases (список из 2-3 фраз на русском)."),
        ("user", "k={k}. archetype_id строго a1..a{k}. все на русском языке.")
    ])

    last_err = None
    for attempt in range(3):
        try:
            raw = llm.invoke(prompt.format_messages(k=k)).content.strip()
            if not raw.startswith("["):
                start = raw.find("[")
                end = raw.rfind("]")
                if start != -1 and end != -1 and end > start:
                    raw = raw[start:end + 1]

            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("not a json list")

            fixed = []
            for item in data[:k]:
                if isinstance(item, dict):
                    item = _ensure_traits(item, rng)
                    fixed.append(Archetype(**item))

            if len(fixed) < k:
                for fb in FALLBACK_ARCHETYPES:
                    if len(fixed) >= k:
                        break
                    if fb.archetype_id not in {a.archetype_id for a in fixed}:
                        fixed.append(fb)

            save_cached_archetypes(fixed, output_dir=output_dir)
            return fixed
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    save_cached_archetypes(FALLBACK_ARCHETYPES[:k], output_dir=output_dir)
    return FALLBACK_ARCHETYPES[:k]
