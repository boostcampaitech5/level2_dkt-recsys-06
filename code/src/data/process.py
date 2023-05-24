import pandas as pd
from .data_modify import age_average_fill_na, average_fill_na, create_feature_test_cat
from sklearn.preprocessing import LabelEncoder
import torch


def index_data(data: dict, settings: dict) -> None:
    """
    Labels data to be used in the embedding layer

    Parameters:
        data(dict): Dictionary containing processed dataframes
        settings(dict): Dictionary containing the settings
    """

    # Saves length of labels per column
    label_len_dict = dict()

    # Loop for each labeling column
    for col in settings["embedding_columns"]:
        # Create label encoder and fit unique values
        le = LabelEncoder()
        unique_value = data["train"][col].unique().tolist() + ["unknown"]
        le.fit(unique_value)

        # Change test dataset to fit labels
        # If the label doesn't exist then set label to unknown
        data["test"][col] = data["test"][col].apply(
            lambda x: x if str(x) in le.classes_ else "unknown"
        )

        # Map the labels
        data["train"][col] = le.transform(data["train"][col])
        data["test"][col] = le.transform(data["test"][col])

        # Save the length of label
        label_len_dict[col] = len(unique_value)

        settings["label_len_dict"] = label_len_dict

    return


def process_lstm(data: dict) -> None:
    """
    Processes data for the LSTM model

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes
    """

    # Order data by user and time
    data["train"] = data["train"].sort_values(by=["user_id", "timestamp"], axis=0)
    data["test"] = data["test"].sort_values(by=["user_id", "timestamp"], axis=0)

    # Create a feature called test_cat
    # create_feature_test_cat(data)

    return


def process_lstm_attn(data) -> None:
    """
    Processes data for the LSTM attention model

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes
    """

    # Order data by user and time
    data["train"] = data["train"].sort_values(by=["user_id", "timestamp"], axis=0)
    data["test"] = data["test"].sort_values(by=["user_id", "timestamp"], axis=0)

    # Create a feature called test_cat
    # create_feature_test_cat(data)

    return


def process_bert(data) -> None:
    """
    Processes data for the BERT model

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes
    """

    # Order data by user and time
    data["train"] = data["train"].sort_values(by=["user_id", "timestamp"], axis=0)
    data["test"] = data["test"].sort_values(by=["user_id", "timestamp"], axis=0)

    # Create a feature called test_cat
    # create_feature_test_cat(data)

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
        subset=["user_id", "question_id"], keep="last", inplace=True
    )

    # Training data, test data reclassification = separate_data
    data["train"] = data["concat_data"][data["concat_data"]["answer_code"] >= 0]
    data["test"] = data["concat_data"][data["concat_data"]["answer_code"] < 0]

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
        sorted(data_concat["user_id"].unique().tolist()),
        sorted(data_concat["question_id"].unique().tolist()),
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
        data["user_id"], data["question_id"], data["answer_code"]
    ):
        edges.append([node2idx[user_id], node2idx[item_id]])
        labels.append(answer_code)

    # Convert to tensor
    edges = torch.LongTensor(edges).T
    labels = torch.LongTensor(labels)

    return dict(edge=edges.to(device), label=labels.to(device))


def process_data(data: dict, settings: dict, silence=False) -> None:
    """
    Merges / Drops columns / Indexes from data in order

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes
        settings(dict): Dictionary containing the settings
    """
    if silence:
        global print
        print = str
    # Modify data
    print("Modifing Data...")

    # Modify/Create columns in data
    if settings["model_name"].lower() == "lstm":
        process_lstm(data)
    elif settings["model_name"].lower() == "lstm_attn":
        process_lstm_attn(data)
    elif settings["model_name"].lower() == "bert":
        process_bert(data)
    elif settings["model_name"].lower() == "lgcn":
        process_lgcn(data)
        print("Make Node Information!!...")
        node2idx = make_nodes(data["concat_data"])
        data["num_nodes"] = len(node2idx)
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
    data["train"] = data["train"][settings["train_columns"]]
    data["test"] = data["test"][settings["train_columns"]]

    print("Dropped Columns!")
    print()

    print("Indexing Columns...")

    # Label columns
    index_data(data, settings)

    print("Indexed Columns!")
    print()

    return
