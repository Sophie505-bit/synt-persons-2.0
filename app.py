import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Синтетические респонденты", layout="wide")

# ── sidebar ──
st.sidebar.title("Настройки")
api_key = st.sidebar.text_input("API key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

base_url = st.sidebar.text_input("Base URL", value=os.environ.get("OPENAI_BASE_URL", ""))
if base_url:
    os.environ["OPENAI_BASE_URL"] = base_url

model_name = st.sidebar.text_input("Model", value=os.environ.get("MODEL_NAME", "gpt-4o-mini"))
if model_name:
    os.environ["MODEL_NAME"] = model_name

data_file = st.sidebar.text_input("Data file", value="set 2/itmo_qa (3) - 8000 респондентов.xlsx")
schema_path = "configs/survey_schema.yaml"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
POPULATION_FILE = os.path.join(DATA_DIR, "synthetic_population.jsonl")
SURVEY_FILE = os.path.join(DATA_DIR, "survey_results.json")
SCENARIO_FILE = os.path.join(DATA_DIR, "scenario_results.json")


# ── helpers ──
@st.cache_data
def load_real_data(path):
    from src.data_loader import DataLoader
    loader = DataLoader(path)
    return loader.get_demographic_summary(), loader.get_baseline_answers()


def load_personas(filepath):
    from src.schema import PersonaProfile
    personas = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    personas.append(PersonaProfile.model_validate_json(line))
    return personas


def load_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_survey_schema():
    from src.schema import load_schema
    return load_schema(schema_path)


# ── tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📊 Реальные данные", "🧬 Генерация", "📋 Опрос", "📈 Оценка"])

# ── TAB 1: real data ──
with tab1:
    st.header("Реальные данные")
    if not os.path.exists(data_file):
        st.warning(f"Файл {data_file} не найден")
    else:
        summary, _ = load_real_data(data_file)
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего респондентов", summary["total"])
        c2.metric("Средний возраст", f"{summary['age_mean']:.1f}")
        c3.metric("Ст. откл. возраста", f"{summary['age_std']:.1f}")

        gender_df = pd.DataFrame(list(summary["gender_dist"].items()), columns=["Пол", "Кол-во"])
        st.plotly_chart(px.pie(gender_df, values="Кол-во", names="Пол", title="Распределение по полу"),
                        use_container_width=True)

        st.subheader("Распределение по доходу")
        inc_df = pd.DataFrame(list(summary["income_dist"].items()), columns=["Доход", "Кол-во"])
        st.dataframe(inc_df, use_container_width=True)

# ── TAB 2: generate ──
with tab2:
    st.header("Генерация популяции")
    n_personas = st.slider("Количество персон", 10, 1000, 100, 10)

    if st.button("🚀 Генерировать", type="primary"):
        if not os.path.exists(data_file):
            st.error(f"Файл данных не найден: {data_file}")
        else:
            # удаляем старый кэш английских архетипов
            cache_path = os.path.join(DATA_DIR, "archetypes_cache.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as _f:
                        _test = json.load(_f)
                    if _test and isinstance(_test[0].get("name", ""), str) and _test[0]["name"].isascii():
                        os.remove(cache_path)
                        st.info("Удалён старый кэш архетипов (был на английском)")
                except Exception:
                    pass

            from src.data_loader import DataLoader
            from src.generator import generate_population_archetypes

            loader = DataLoader(data_file)
            df = loader.get_joint_distribution()
            progress = st.progress(0)
            status = st.empty()

            def on_progress(cur, total, p):
                progress.progress(cur / total)
                status.text(f"Генерация: {cur}/{total}")

            with st.spinner("Генерация популяции..."):
                results = generate_population_archetypes(
                    df, n=n_personas, output_file=POPULATION_FILE,
                    progress_callback=on_progress
                )
            st.success(f"Создано {len(results)} профилей ✓")

    personas = load_personas(POPULATION_FILE)
    if personas:
        st.subheader(f"Загружено {len(personas)} профилей")
        demo_data = [{
            "id": p.persona_id[:8],
            "возраст": p.demographics.age,
            "пол": p.demographics.gender,
            "район": p.demographics.region[:20],
            "архетип": p.archetype_id,
        } for p in personas[:200]]
        st.dataframe(pd.DataFrame(demo_data), height=300)

        st.subheader("Примеры нарративов")
        for p in personas[:3]:
            with st.expander(f"{p.demographics.gender}, {p.demographics.age} лет, {p.archetype_id}"):
                st.write(p.narrative)
                st.json(p.behaviors)

# ── TAB 3: survey ──
with tab3:
    st.header("Опрос")
    personas = load_personas(POPULATION_FILE)
    if not personas:
        st.warning("Сначала сгенерируйте популяцию (вкладка Генерация)")
    else:
        n_survey = st.slider("Сколько опросить", 5, min(len(personas), 500), min(50, len(personas)), 5)

        use_scenario = st.checkbox("Использовать сценарий")
        scenario_text = ""
        if use_scenario:
            default_scenario = ""
            if os.path.exists("data/scenario.txt"):
                with open("data/scenario.txt", "r", encoding="utf-8") as f:
                    default_scenario = f.read()
            scenario_text = st.text_area("Сценарий", value=default_scenario, height=100)

        output_file = SCENARIO_FILE if use_scenario else SURVEY_FILE

        if st.button("▶️ Запустить опрос", type="primary"):
            from src.simulator import Simulator
            from src.priors import compute_real_priors
            from src.graph_builder import build_knn_graph

            _, real_answers = load_real_data(data_file)
            survey_schema = load_survey_schema()
            priors = compute_real_priors(real_answers, survey_schema)

            # граф соседей
            selected_personas = personas[:n_survey]
            embeddings = [p.embedding for p in selected_personas if p.embedding]
            neighbors = {}
            if len(embeddings) == len(selected_personas):
                neighbors = build_knn_graph(embeddings, k=min(10, len(selected_personas) - 1))

            sim = Simulator(schema_path, priors=priors)
            progress = st.progress(0)
            status = st.empty()

            def on_survey_progress(cur, total):
                progress.progress(cur / total)
                status.text(f"Опрос: {cur}/{total}")

            with st.spinner("Опрос идёт..."):
                results = sim.run_survey(
                    selected_personas,
                    scenario=scenario_text,
                    neighbors_graph=neighbors,
                    progress_callback=on_survey_progress,
                )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            st.success(f"Опрошено {len(results)} персон → {output_file} ✓")

        # отображение результатов
        results = load_results(output_file)
        if results:
            st.subheader(f"Результаты ({len(results)} ответов)")
            survey_schema = load_survey_schema()
            for q in survey_schema.questions:
                option_counts = {opt.id: 0 for opt in q.options}
                for r in results:
                    for sid in r.get(q.question_id, []):
                        if sid in option_counts:
                            option_counts[sid] += 1
                chart_data = pd.DataFrame([
                    {"вариант": f"{opt.id}", "доля": option_counts[opt.id] / len(results)}
                    for opt in q.options
                ])
                fig = px.bar(chart_data, x="доля", y="вариант", orientation="h",
                             title=f"{q.question_id}: {q.text[:70]}...",
                             height=max(250, len(q.options) * 28))
                fig.update_layout(xaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: evaluate ──
with tab4:
    st.header("Оценка качества")

    compare_mode = st.radio("Режим сравнения", ["Real vs Synthetic", "Baseline vs Scenario"])

    if compare_mode == "Real vs Synthetic":
        results = load_results(SURVEY_FILE)
        if not results:
            st.warning("Сначала запустите опрос без сценария")
        elif not os.path.exists(data_file):
            st.warning("Файл данных не найден")
        else:
            from src.evaluator import Evaluator
            _, real_answers = load_real_data(data_file)
            ev = Evaluator(schema_path)
            real_agg = ev.aggregate_answers(real_answers)
            synth_agg = ev.aggregate_answers(results)
            survey_schema = load_survey_schema()

            metrics = []
            for q in survey_schema.questions:
                qid = q.question_id
                option_ids = [opt.id for opt in q.options]
                tvd = ev.compute_tvd(real_agg.get(qid, {}), synth_agg.get(qid, {}), option_ids)
                metrics.append({"вопрос": qid, "TVD": f"{tvd:.3f}"})

                real_vals = [real_agg.get(qid, {}).get(oid, 0) for oid in option_ids]
                synth_vals = [synth_agg.get(qid, {}).get(oid, 0) for oid in option_ids]
                fig = go.Figure()
                fig.add_trace(go.Bar(name="реальные", x=option_ids, y=real_vals, marker_color="#4C72B0"))
                fig.add_trace(go.Bar(name="синтетические", x=option_ids, y=synth_vals, marker_color="#DD8452"))
                fig.update_layout(title=f"{qid} — TVD: {tvd:.3f}", barmode="group",
                                  yaxis_range=[0, 1], height=350)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Сводка TVD")
            st.table(pd.DataFrame(metrics))

    else:
        base_results = load_results(SURVEY_FILE)
        scen_results = load_results(SCENARIO_FILE)
        if not base_results:
            st.warning("Сначала запустите baseline опрос (без сценария)")
        elif not scen_results:
            st.warning("Запустите опрос со сценарием")
        else:
            from src.evaluator import Evaluator
            ev = Evaluator(schema_path)
            base_agg = ev.aggregate_answers(base_results)
            scen_agg = ev.aggregate_answers(scen_results)
            survey_schema = load_survey_schema()

            st.subheader("Влияние сценария")
            for q in survey_schema.questions:
                qid = q.question_id
                option_ids = [opt.id for opt in q.options]
                tvd = ev.compute_tvd(base_agg.get(qid, {}), scen_agg.get(qid, {}), option_ids)

                base_vals = [base_agg.get(qid, {}).get(oid, 0) for oid in option_ids]
                scen_vals = [scen_agg.get(qid, {}).get(oid, 0) for oid in option_ids]
                diff_vals = [s - b for s, b in zip(scen_vals, base_vals)]

                fig = go.Figure()
                fig.add_trace(go.Bar(name="baseline", x=option_ids, y=base_vals, marker_color="#4C72B0"))
                fig.add_trace(go.Bar(name="scenario", x=option_ids, y=scen_vals, marker_color="#C44E52"))
                fig.update_layout(title=f"{qid} — TVD: {tvd:.3f}", barmode="group",
                                  yaxis_range=[0, 1], height=350)
                st.plotly_chart(fig, use_container_width=True)

                colors = ["green" if d >= 0 else "red" for d in diff_vals]
                fig_delta = go.Figure()
                fig_delta.add_trace(go.Bar(x=option_ids, y=diff_vals, marker_color=colors))
                fig_delta.update_layout(title=f"{qid} — дельта (сценарий − baseline)", height=250)
                st.plotly_chart(fig_delta, use_container_width=True)
