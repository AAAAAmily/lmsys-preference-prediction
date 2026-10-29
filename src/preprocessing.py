import json
import numpy as np


def parse_text_fields(df):
    df = df.copy()

    for col in ["prompt", "response_a", "response_b"]:
        df[f"{col}_list"] = df[col].apply(json.loads)

    return df


def create_label(df):
    df = df.copy()

    conditions = [
        df["winner_model_a"] == 1,
        df["winner_model_b"] == 1,
        df["winner_tie"] == 1
    ]

    choices = ["A", "B", "tie"]

    df["label"] = np.select(
        conditions,
        choices,
        default=None
    )

    return df


def preprocess_data(df):
    df = parse_text_fields(df)
    df = create_label(df)

    return df