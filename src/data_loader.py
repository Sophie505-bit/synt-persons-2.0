import pandas as pd
from typing import List, Dict, Any
from src.schema import DemographicProfile


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)

    def get_raw_data(self) -> pd.DataFrame:
        return self.df

    def get_demographics(self) -> List[DemographicProfile]:
        profiles = []
        for _, row in self.df.iterrows():
            profiles.append(DemographicProfile(
                age=int(row["Возраст"]),
                gender=str(row["Пол"]),
                region=str(row["Район проживания"]),
                income=str(row["Вопрос 5"]),
                education=str(row["Вопрос 6"]),
            ))
        return profiles

    def get_baseline_answers(self) -> List[Dict[str, Any]]:
        answers = []
        for _, row in self.df.iterrows():
            ans = {}
            for prefix, qid in [("1.", "q1"), ("2.", "q2"), ("3.", "q3"), ("4.", "q4")]:
                selected = []
                for col in self.df.columns:
                    if col.startswith(prefix) and row[col] == 1:
                        selected.append(col)
                ans[qid] = selected
            answers.append(ans)
        return answers

    def get_joint_distribution(self) -> pd.DataFrame:
        return self.df[["Возраст", "Пол", "Район проживания", "Вопрос 5", "Вопрос 6"]].copy()

    def get_demographic_summary(self) -> Dict[str, Any]:
        df = self.df
        return {
            "total": len(df),
            "age_mean": df["Возраст"].mean(),
            "age_std": df["Возраст"].std(),
            "gender_dist": df["Пол"].value_counts().to_dict(),
            "region_dist": df["Район проживания"].value_counts().to_dict(),
            "income_dist": df["Вопрос 5"].value_counts().to_dict(),
            "education_dist": df["Вопрос 6"].value_counts().to_dict(),
        }
