import pandas as pd


def create_test_mean_sum(data: pd.DataFrame) -> None:
    correct_t_train = (
        data["train"].groupby(["test_id"])["answer_code"].agg(["mean", "sum"])
    )
    correct_t_train.columns = ["test_mean", "test_sum"]
    data["train"] = pd.merge(data["train"], correct_t_train, on=["test_id"], how="left")

    correct_t_test = (
        data["test"].groupby(["test_id"])["answer_code"].agg(["mean", "sum"])
    )
    correct_t_test.columns = ["test_mean", "test_sum"]
    data["test"] = pd.merge(data["test"], correct_t_train, on=["test_id"], how="left")

    return
