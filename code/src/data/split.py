import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, settings: dict):
        """
        Initializes DKTDataset

        Parameters:
            data(np.ndarray): Numpy array of processed data
            settings(dict): Dictionary containing the settings
        """

        self.data = data

        # Fixed data length
        self.max_seq_len = settings[settings["model_name"].lower()]["max_seq_len"]

        # The column that is being predicted
        self.predict_column = settings["predict_column"]

        # The indexed columns
        self.index_column = settings["embedding_columns"]

    def __getitem__(self, index: int) -> dict:
        """
        Gets data in index
        Processes the data to return a dict with each column as the index

        Parameters:
            data(np.ndarray): Numpy array of processed data
            settings(dict): Dictionary containing the settings

        Returns:
            row_data(dict): Dictionary containing the data divided by the column
        """

        # Load from data
        row = self.data[index]

        # Load from data
        row_data = {
            i: torch.tensor(v + 1, dtype=torch.int)
            for i, v in row.items()
            if i in self.index_column
        }

        # Leave the predict column alone
        row_data[self.predict_column] = torch.tensor(
            row[self.predict_column], dtype=torch.int
        )

        # Generate mask
        seq_len = len(list(row.values())[0])

        # Pad data
        if seq_len > self.max_seq_len:
            for k, seq in row_data.items():
                row_data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in row_data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len - seq_len :] = row_data[k]
                row_data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1

        # Save mask
        row_data["mask"] = mask

        # Generate interaction
        interaction = row_data[self.predict_column] + 1
        interaction = interaction.roll(shifts=1)
        interaction_mask = row_data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        row_data["interaction"] = interaction
        row_data = {k: v.int() for k, v in row_data.items()}

        return row_data

    def __len__(self) -> int:
        return len(self.data)


def data_split(data: dict, settings: dict) -> None:
    """
    Splits train data to train data and validation data

    Parameters:
        data(dict): Dictionary containing the processed data
        settings(dict): Dictionary containing the settings
    """

    print("Splitting dataset...")

    if not settings["is_graph_model"]:

        def data_split_by_seq(input_df: pd.DataFrame):
            input_df = input_df.iloc[: settings["max_train_length"]]
            return input_df.groupby(
                np.flip(np.arange(len(input_df.index)))
                // settings[settings["model_name"].lower()]["max_seq_len"]
            ).apply(lambda x: {c: x[c].values for c in column_list if c != "user_id"})

        # Group by user and combine all columns
        column_list = data["train"].columns
        if settings["extra_split"]:
            data["train"] = data["train"].groupby("user_id").apply(data_split_by_seq)
        else:
            data["train"] = (
                data["train"]
                .groupby("user_id")
                .apply(
                    lambda x: {c: x[c].values for c in column_list if c != "user_id"}
                )
            )
        data["test"] = (
            data["test"]
            .groupby("user_id")
            .apply(lambda x: {c: x[c].values for c in column_list if c != "user_id"})
        )

        # Change data to numpy arrays
        data["train"] = data["train"].to_numpy()
        data["test"] = data["test"].to_numpy()

        # Fix to default seed 0
        # This is needed when ensembling both files
        ## Having both files with different train and valid datasets will prevent us from making a loss estimate
        random.seed(0)

        # Shuffle the train dataset randomly
        random.shuffle(data["train"])

        # Divide data by ratio
        train_size = int(len(data["train"]) * settings["train_valid_split"])
        data["valid"] = data["train"][train_size:]
        data["train"] = data["train"][:train_size]

    else:
        np.random.seed(0)

        # size : # of edge of train
        size_all = len(data["train"]["label"])
        train_size = int(size_all * settings["train_valid_split"])

        edge_ids = np.arange(size_all)
        edge_ids = np.random.permutation(edge_ids)

        train_edge_ids = edge_ids[:train_size]
        valid_edge_ids = edge_ids[train_size:]

        edge, label = data["train"]["edge"], data["train"]["label"]
        # label = label.to("cpu").detach().numpy()

        data["train"] = dict(edge=edge[:, train_edge_ids], label=label[train_edge_ids])
        data["valid"] = dict(edge=edge[:, valid_edge_ids], label=label[valid_edge_ids])

    print("Splitted Data!")
    print()

    return


def create_datasets(data: dict, settings: dict) -> dict:
    """
    Creates datasets using the train, valid, test data

    Parameters:
        data(dict): Dictionary containing processed dataframes
        settings(dict): Dictionary containing the settings

    Returns:
        dataset(dict): Dictionary containing loaded datasets
    """

    # For graph_based models, omit this function
    if settings["is_graph_model"]:
        dataset = {"train": data["train"], "valid": data["valid"], "test": data["test"]}
        return dataset

    print("Creating Datasets!")

    dataset = dict()

    # Create datasets using class
    dataset["train"] = DKTDataset(data["train"], settings)
    dataset["valid"] = DKTDataset(data["valid"], settings)
    dataset["test"] = DKTDataset(data["test"], settings)

    print("Created Datasets!")
    print()

    return dataset


def create_dataloader(dataset: dict, settings: dict) -> dict:
    """
    Creates dataloader from datasets.

    Parameters:
        dataset(dict): Dictionary containing datasets.

    Returns:
        dataloader(dict): Dictionary loaded dataloaders.
    """

    print("Creating Dataloader...")

    dataloader = dict()
    if not settings["is_graph_model"]:
        # Create dataloader
        dataloader["train"] = DataLoader(
            dataset["train"],
            batch_size=settings["batch_size"],
            shuffle=True,
            num_workers=settings["num_workers"],
            pin_memory=False,
        )
        dataloader["valid"] = DataLoader(
            dataset["valid"],
            batch_size=settings["batch_size"],
            shuffle=False,
            num_workers=settings["num_workers"],
            pin_memory=False,
        )
        dataloader["test"] = DataLoader(
            dataset["test"],
            batch_size=settings["batch_size"],
            shuffle=False,
            num_workers=settings["num_workers"],
            pin_memory=False,
        )
    else:
        dataloader["train"] = dataset["train"]
        dataloader["valid"] = dataset["valid"]
        dataloader["test"] = dataset["test"]

    print("Created Dataloader!")
    print()

    return dataloader
