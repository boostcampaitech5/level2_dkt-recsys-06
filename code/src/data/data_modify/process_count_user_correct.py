import pandas as pd


def count_users_correct_answer(data: pd.DataFrame) -> None:
    data["train"]["user_correct_answer"] = (
        data["train"]
        .groupby("user_id")["answer_code"]
        .transform(lambda x: x.cumsum().shift(1))
    )
    data["test"]["user_correct_answer"] = (
        data["test"]
        .groupby("user_id")["answer_code"]
        .transform(lambda x: x.cumsum().shift(1))
    )

    return
