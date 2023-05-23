import pandas as pd


def create_feature_test_solved_num(data: dict) -> None:
    data["train"]["test_solved_num"] = (
        data["train"]
        .sort_values(["timestamp"])
        .groupby(["user_id", "test_numeric_id"])
        .cumcount()
    )
    data["test"]["test_solved_num"] = (
        data["train"]
        .sort_values(["timestamp"])
        .groupby(["user_id", "test_numeric_id"])
        .cumcount()
    )

    return


def create_feature_user_solved_num(data: dict) -> None:
    data["train"]["user_solved_num"] = (
        data["train"].sort_values(["timestamp"]).groupby(["user_id"]).cumcount()
    )
    data["test"]["user_solved_num"] = (
        data["train"].sort_values(["timestamp"]).groupby(["user_id"]).cumcount()
    )

    return
