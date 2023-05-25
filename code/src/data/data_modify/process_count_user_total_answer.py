import pandas as pd


def count_users_total_answer(data: pd.DataFrame) -> None:
    data["train"]["user_total_answer"] = (
        data["train"].groupby("user_id")["answer_code"].cumcount()
    )
    data["test"]["user_total_answer"] = (
        data["test"].groupby("user_id")["answer_code"].cumcount()
    )

    return
