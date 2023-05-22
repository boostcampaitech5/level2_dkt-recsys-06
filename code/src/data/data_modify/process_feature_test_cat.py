import pandas as pd


def create_feature_test_cat(data: pd.DataFrame) -> None:
    data["train"]["test_cat"] = data["train"]["question_id"].apply(lambda x: x[2])
    data["test"]["test_cat"] = data["test"]["question_id"].apply(lambda x: x[2])

    return
