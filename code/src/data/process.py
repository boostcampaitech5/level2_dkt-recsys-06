import pandas as pd
from .data_modify import age_average_fill_na, average_fill_na, create_feature_test_cat
from sklearn.preprocessing import LabelEncoder


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

    # Save the dictionary in the settings dictionary
    settings["label_len_dict"] = label_len_dict

    return


def process_mlp(data: dict) -> None:
    """
    Processes data for the MLP model

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes
    """

    average_fill_na(data["user_data"], "age")

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
    create_feature_test_cat(data)

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
    create_feature_test_cat(data)

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
    create_feature_test_cat(data)

    return


def process_data(data: dict, settings: dict) -> None:
    """
    Merges / Drops columns / Indexes from data in order

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes
        settings(dict): Dictionary containing the settings
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
