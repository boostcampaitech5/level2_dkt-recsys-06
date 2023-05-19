import pandas as pd

# from .data_modify import age_average_fill_na, average_fill_na, create_feature_big_tag
from sklearn.preprocessing import LabelEncoder
import torch


def index_data(data: dict, settings: dict) -> dict:
    """
    Labels data to be used in embedding layers

    Parameters:
        data(dict): Dictionary containing processed dataframes.
        settings(dict): Dictionary containing the settings.

    Returns:
        idx(dict): Dictionary containing the length of each columns
                   Used in making embedded layers
    """
    idx = dict()

    # Loop for each indexing columns
    for index_column in settings["index_columns"]:
        # Create label encoder and fit values to it
        le = LabelEncoder()
        unique_value = data["train"][index_column].unique().tolist() + ["unknown"]
        le.fit(unique_value)

        # Change test dataset to fit labels
        data["test"][index_column] = data["test"][index_column].apply(
            lambda x: x if str(x) in le.classes_ else "unknown"
        )

        # Map the labels to values
        data["train"][index_column] = le.transform(data["train"][index_column])
        data["test"][index_column] = le.transform(data["test"][index_column])

        # Save length of label for future use
        idx[index_column] = len(unique_value)

    return idx


def process_mlp(data: dict) -> None:
    """
    Processes data for MLP training.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
    """

    average_fill_na(data["user_data"], "age")

    return


def process_lstm(data: dict) -> None:
    # Order data by user and time
    data["train"] = data["train"].sort_values(by=["userID", "Timestamp"], axis=0)
    data["test"] = data["test"].sort_values(by=["userID", "Timestamp"], axis=0)

    create_feature_big_tag(data)

    return


def process_lstm_attn(data) -> None:
    # Order data by user and time
    data["train"] = data["train"].sort_values(by=["userID", "Timestamp"], axis=0)
    data["test"] = data["test"].sort_values(by=["userID", "Timestamp"], axis=0)

    create_feature_big_tag(data)

    return


def process_bert(data) -> None:
    # Order data by user and time
    data["train"] = data["train"].sort_values(by=["userID", "Timestamp"], axis=0)
    data["test"] = data["test"].sort_values(by=["userID", "Timestamp"], axis=0)

    create_feature_big_tag(data)

    return


def process_lgcn(data: dict) -> None:
    """Append merge data : train + test & Split train and test

    Args:
        data (dict): data .. key : train, test
    """
    # concatenate 'train data set & Test data set
    data["concat_data"] = pd.concat([data["train"], data["test"]])

    # Among duplicate user_id and assessmentItemID, only the last one is deleted.
    data["concat_data"].drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    # Training data, test data reclassification = separate_data
    data["train"] = data["concat_data"][data["concat_data"]["answerCode"] >= 0]
    data["test"] = data["concat_data"][data["concat_data"]["answerCode"] < 0]

    return


def make_nodes(data_concat: pd.DataFrame) -> dict:
    """This function creates node information.
       Node information includes user ID and item ID.

    Args:
        data_concat (pd.DataFrame): Combined training and test data.

    Returns:
        dict: node_to_idx
    """
    user_id, item_id = (
        sorted(data_concat["userID"].unique().tolist()),
        sorted(data_concat["assessmentItemID"].unique().tolist()),
    )
    # merge user_id & item_id
    node_id = user_id + item_id

    # Initialization dictionary : node2idx
    node2idx = {node: idx for idx, node in enumerate(node_id)}

    return node2idx


def get_edge_label_dict(data: pd.DataFrame, node2idx: dict, device: str) -> dict:
    """Create a dictionary of edges and labels

    Args:
        data (pd.DataFrame): train data or test data
        node2idx (dict): {node:index} dictionary
        device (str): cuda or cpu

    Returns:
        dict: Return {edge: edges, label: labels}, edges,labels:list
    """
    edges, labels = [], []

    # Label Encoding & Append edge and label
    for user_id, item_id, answer_code in zip(
        data["userID"], data["assessmentItemID"], data["answerCode"]
    ):
        edges.append([node2idx[user_id], node2idx[item_id]])
        labels.append(answer_code)

    # Convert to tensor
    edges = torch.LongTensor(edges).T
    labels = torch.LongTensor(labels)

    return dict(edge=edges.to(device), label=labels.to(device))


def process_data(data: dict, settings: dict) -> None:
    """
    Merges / Drops columns / Indexes from data.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """
    # Modify data
    print("Modifing Data...")

    # Modify/Create columns in data
    if settings["model_name"].lower() == "mlp":
        process_mlp(data)
    elif settings["model_name"].lower() == "lstm":
        process_lstm(data)
    elif settings["model_name"].lower() == "lstm_attn":
        process_lstm_attn(data)
    elif settings["model_name"].lower() == "bert":
        process_bert(data)
    elif settings["model_name"].lower() == "lgcn":
        process_lgcn(data)
        print("Make Node Information!!...")
        node2idx = make_nodes(data["concat_data"])
        data["n_node"] = len(node2idx)
        data["train"] = get_edge_label_dict(
            data["train"], node2idx=node2idx, device=settings["device"]
        )
        data["test"] = get_edge_label_dict(
            data["test"], node2idx=node2idx, device=settings["device"]
        )
        return

    else:
        print("Found no processing function...")
        print("Not processing any data...")

    print("Modified Data!")
    print()

    print("Dropping Columns...")

    # Drop unwanted columns
    data["train"] = data["train"][settings["choose_columns"]]
    data["test"] = data["test"][settings["choose_columns"]]

    print("Dropped Columns!")
    print()

    print("Indexing Columns...")

    # Label columns
    data["idx"] = index_data(data, settings)

    print("Indexed Columns!")
    print()

    return


if __name__ == "__main__":
    data_path = "/opt/ml/input/data/"
    data = dict()
    data["train"] = pd.read_csv(data_path + "train_data.csv")
    data["test"] = pd.read_csv(data_path + "test_data.csv")

    print(f"train data 개수 : {len(data['train'])}")
    # print("*******************train_head*******************")
    # print(data['train'].head())
    # print("*******************train_tail*******************")
    # print(data['train'].tail())

    print("\n\n")

    print(f"test data 개수: {len(data['test'])}")
    # print("*******************test_head*******************")
    # print(data['test'].head())
    # print("*******************test_tail*******************")
    # print(data['test'].tail())

    print("\n\n")

    process_lgcn(data)

    print("*********************************************************")
    print("**********************process_lgcn***********************")

    print(f"concat data 개수: {len(data['concat_data'])}")
    # print("*******************concat_head*******************")
    # print(data['concat_data'].head())
    # print("*******************concat_tail*******************")
    # print(data['concat_data'].tail())

    print("\n\n")

    print(f"train data 개수 : {len(data['train'])}")
    # print("*******************train_head*******************")
    # print(data['train'].head())
    # print("*******************train_tail*******************")
    # print(data['train'].tail())

    print("\n\n")

    print(f"test data 개수: {len(data['test'])}")
    # print("*******************test_head*******************")
    # print(data['test'].head())
    # print("*******************test_tail*******************")
    # print(data['test'].tail())

    node2idx = make_nodes(data_concat=data["concat_data"])

    print(f"node의 개수 : {len(node2idx)}")
    print(list(node2idx.items())[-1])

    data["train"] = get_edge_label_dict(data["train"], node2idx=node2idx, device="cpu")
    data["test"] = get_edge_label_dict(data["test"], node2idx=node2idx, device="cpu")

    print(f"[train] The number : {len(data['train']['edge'][0])}")
    print(data["train"]["edge"][0][:5])
    print(data["train"]["edge"][1][:5])
    print(data["train"]["label"][:5])

    print(data["train"]["edge"][0][-5:])
    print(data["train"]["edge"][1][-5:])
    print(data["train"]["label"][-5:])

    print(f"[test] The number : {len(data['test']['edge'][0])}")
    print(data["test"]["edge"][0][:5])
    print(data["test"]["edge"][1][:5])
    print(data["test"]["label"][:5])

    print(data["test"]["edge"][0][-5:])
    print(data["test"]["edge"][1][-5:])
    print(data["test"]["label"][-5:])
