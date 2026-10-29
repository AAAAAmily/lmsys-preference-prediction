import numpy as np


def total_char_length(turns):
    return sum(
        len(turn)
        for turn in turns
        if isinstance(turn, str)
    )


def add_structural_features(df):
    df = df.copy()

    # response length
    df["a_chars"] = df["response_a_list"].apply(total_char_length)
    df["b_chars"] = df["response_b_list"].apply(total_char_length)

    df["length_diff"] = df["a_chars"] - df["b_chars"]
    df["abs_length_diff"] = df["length_diff"].abs()

    df["log_length_ratio"] = np.log(
        (df["a_chars"] + 1) /
        (df["b_chars"] + 1)
    )

    df["normalized_length_diff"] = (
        df["a_chars"] - df["b_chars"]
    ) / (
        df["a_chars"] + df["b_chars"] + 1
    )

    # turn structure
    df["prompt_turns"] = df["prompt_list"].apply(len)
    df["is_multi_turn"] = (df["prompt_turns"] > 1).astype(int)

    return df