import pandas as pd
from .data_modify import age_average_fill_na, average_fill_na, create_feature_big_tag
from sklearn.preprocessing import LabelEncoder


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
    max_label_dict = dict()

    # Loop for each indexing columns
    for col in settings["embedding_columns"]:
        # Create label encoder and fit values to it
        le = LabelEncoder()
        unique_value = data["train"][col].unique().tolist() + ["unknown"]
        le.fit(unique_value)

        # Change test dataset to fit labels
        data["test"][col] = data["test"][col].apply(
            lambda x: x if str(x) in le.classes_ else "unknown"
        )

        # Map the labels to values
        data["train"][col] = le.transform(data["train"][col])
        data["test"][col] = le.transform(data["test"][col])

        # Save length of label for future use
        max_label_dict[col] = len(unique_value)

    settings["max_label_dict"] = max_label_dict

    return


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
