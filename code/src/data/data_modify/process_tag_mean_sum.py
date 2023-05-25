import pandas as pd


def create_tag_mean_sum(data: pd.DataFrame) -> None:
    correct_k_train = (
        data["train"].groupby(["knowledge_tag"])["answer_code"].agg(["mean", "sum"])
    )
    correct_k_train.columns = ["tag_mean", "tag_sum"]
    data["train"] = pd.merge(
        data["train"], correct_k_train, on=["knowledge_tag"], how="left"
    )

    correct_k_test = (
        data["test"].groupby(["knowledge_tag"])["answer_code"].agg(["mean", "sum"])
    )
    correct_k_test.columns = ["tag_mean", "tag_sum"]
    data["test"] = pd.merge(
        data["test"], correct_k_test, on=["knowledge_tag"], how="left"
    )

    return
