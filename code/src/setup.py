import os
import json
import random
from datetime import datetime
from typing import Type

import numpy as np
import pandas as pd
import torch
import wandb


# Settings file name
SETTING_FILE = "setting.json"


def get_general_setting(folder_path: str) -> dict:
    """
    Returns the setting.json file as a dictionary

    Parameters:
        folder_path(str): String containing the path to the project folder

    Returns:
        settings(dict): Dictionary containing the settings
    """

    # Open json file
    with open(os.path.join(folder_path, SETTING_FILE)) as f:
        try:
            # Load file
            settings = json.load(f)
        except:
            # If file not found
            print("Running code in wrong folder.")
            print("Run python file in ModelRunningTemplate folder")
            return 1

    return settings


def set_basic_settings(settings: dict) -> None:
    """
    Setup basic settings

    Parameters:
        settings(dict): Dictionary containing the settings
    """

    # Check if GPU is available
    if not torch.cuda.is_available() and settings["cuda"] == "cuda":
        print("Cuda not Found")
        print("Setting Device to CPU")

        # If not change device to cpu
        settings["device"] = "cpu"

    settings["save_name"] = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        if settings["save_name"] == ""
        else settings["save_name"]
    )

    # Get seed from settings
    seed = settings["seed"]

    # Is graph?
    settings["is_graph_model"] = settings["model_name"] == "lgcn"

    if settings["is_graph_model"]:
        # Set parameter of model:lgcn alpha
        settings["lgcn"]["alpha"] = (
            None if settings["lgcn"]["alpha"] == 0 else settings["lgcn"]["alpha"]
        )
        # Set weight decay = 0
        settings["adam"]["weight_decay"] = 0

    # Apply settings to all randomization
    # All results will be fixed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Get list of non-embedding columns
    settings["non_embedding_columns"] = list(
        set(settings["selective_train_columns"])
        - set(settings["selective_embedding_columns"])
    )

    # Get list of training columns
    settings["train_columns"] = (
        settings["selective_train_columns"] + settings["fixed_train_columns"]
    )

    # Get list of embedding columns
    settings["embedding_columns"] = (
        settings["selective_embedding_columns"] + settings["fixed_embedding_columns"]
    )

    return


def get_unprocessed_data(folder_path: str, settings: dict) -> dict:
    """
    Gets unprocessed train/test data as dataframe and returns it in a dictionary of dataframes

    Parameters:
        folder_path(str): String containing the path to the project folder
        settings(dict): Dictionary containing the settings

    Returns:
        data(dict): Dictionary containing the unprocessed dataframes
    """

    # Used to save unprocessed dataframes
    data = dict()

    # Save path to data file
    data_path = os.path.join(folder_path, settings["path"]["data"])

    # Loop through file names
    for index_name, file_name in settings["file_name"].items():
        # Get raw data files as dataframe and save it in the data dict
        data[index_name] = pd.read_csv(os.path.join(data_path, file_name + ".csv"))

    return data


class SaveSetting:
    """
    Used to control the saving of files and logs
    """

    def __init__(self, folder_path: str, settings: dict, sid: int):
        """
        Initializes SaveSetting class

        Parameters:
            folder_path(str): String containing the path to the project folder
            settings(dict): Dictionary containing the settings
            sid(int): Unique id for save settings
        """

        # Setup folder paths
        self.log_folder_path = os.path.join(folder_path, settings["path"]["log"])
        self.model_folder_path = os.path.join(folder_path, settings["path"]["model"])
        self.statedict_folder_path = os.path.join(
            folder_path, settings["path"]["state_dict"]
        )
        self.submit_folder_path = os.path.join(folder_path, settings["path"]["submit"])
        self.train_folder_path = os.path.join(folder_path, settings["path"]["train"])
        self.valid_folder_path = os.path.join(folder_path, settings["path"]["valid"])
        self.best_model_folder_path = os.path.join(
            folder_path, settings["path"]["best_model"]
        )

        # File name
        self.name = settings["save_name"] + f"_{sid}"

        # Log file
        self.log_file = None

        # Create missing directories
        self.create_dir()

        # Start logging and write first part of the file
        self.start_log(settings)

    def create_dir(self) -> None:
        """
        Creates missing directories for save locations
        """

        if not os.path.exists(self.log_folder_path):
            os.mkdir(self.log_folder_path)
        if not os.path.exists(self.model_folder_path):
            os.mkdir(self.model_folder_path)
        if not os.path.exists(self.statedict_folder_path):
            os.mkdir(self.statedict_folder_path)
        if not os.path.exists(self.submit_folder_path):
            os.mkdir(self.submit_folder_path)
        if not os.path.exists(self.train_folder_path):
            os.mkdir(self.train_folder_path)
        if not os.path.exists(self.valid_folder_path):
            os.mkdir(self.valid_folder_path)
        if not os.path.exists(self.best_model_folder_path):
            os.mkdir(self.best_model_folder_path)

        return

    def append_log(self, input_str: str) -> None:
        """
        Appends string to log file

        Parameters:
            input_str(str): String that will be appended to the log
        """

        # If log file is not opened open log file and save io
        if self.log_file is None:
            # Get file path
            log_file_path = os.path.join(self.log_folder_path, self.name) + ".txt"

            # Open file
            self.log_file = open(log_file_path, "a")

        # Write string to log file
        self.log_file.writelines(input_str)

        return

    def start_log(self, settings: dict) -> None:
        """
        Starts log and prints pre-determined starting message

        Parameters:
            settings(dict): Dictionary containing the settings
        """

        # Starts writing log as new file
        with open(os.path.join(self.log_folder_path, self.name) + ".txt", "w") as f:
            f.write("Choosen Columns for Training:\t")
            f.write(", ".join(settings["train_columns"]) + "\n")
            f.write("Embedded Columns:\t")
            f.write(", ".join(settings["embedding_columns"]) + "\n")
            f.write("Non-Embedded Columns:\t")
            f.write(", ".join(settings["non_embedding_columns"]) + "\n")
            f.write("=" * 30 + "\n\n")

        return

    def close_log(self) -> None:
        """
        Ends log if log is opened
        """

        # If log is still opened
        if self.log_file is not None:
            # Close log
            self.log_file.close()

        return

    def save_model(self, model) -> None:
        """
        Saves model as file

        Parameters:
            model: Model that is saved
        """

        # Get model path
        model_path = os.path.join(self.model_folder_path, self.name + f"_model.pt")

        # Save model to path
        torch.save(model, model_path)

        return

    def save_best_model(self, model: torch.nn.Module, model_name: str) -> None:
        """Save Best model as file

        Args:
            model (torch.nn.Module): Model that is save
            model_name (str): model_name ex> lstm, lgcn, etc...
        """
        # Get best_model path
        best_model_path = os.path.join(
            self.best_model_folder_path, self.name + model_name + f"_model.pt"
        )

        # Save best_model to path
        torch.save(model, best_model_path)

        return

    def get_best_model_path(self, model_name: str) -> str:
        """Function that returns the path of the best model

        Args:
            model_name (str): model_name ex> lstm, lgcn, etc...

        Returns:
            str: Best Model path
        """
        best_model_path = os.path.join(
            self.best_model_folder_path, self.name + model_name + f"_model.pt"
        )
        return best_model_path

    def save_statedict(
        self,
        model,
        train_final_auc: float,
        train_final_acc: float,
        valid_final_auc: float,
        valid_final_acc: float,
        settings: dict,
    ) -> None:
        """
        Saves model's state dict and extra information

        Parameters:
            model: Model to save state dict
            train_final_auc(float): Final AUC from train data
            train_final_acc(float): Final accuracy from train data
            valid_final_auc(float): Final AUC from valid data
            valid_final_acc(float): Final accuracy from valid data
            settings(dict): Dictionary containing the settings
        """

        # Get path to save input parameters
        state_dict_path = os.path.join(
            self.statedict_folder_path, self.name + f"_statedict"
        )

        # Save all data as a dictionary to path
        torch.save(
            {
                "state_dict": model.state_dict(),
                "train_acc": train_final_acc,
                "train_auc": train_final_auc,
                "valid_acc": valid_final_acc,
                "valid_auc": valid_final_auc,
                "settings": settings,
            },
            state_dict_path,
        )

        return

    def save_submit(self, prediction: list) -> None:
        """
        Saves test result to be submitted as csv

        Parameters:
            prediction(list): test result(y_hat)
        """

        # Create dataframe to save as csv
        submit_df = pd.DataFrame(prediction, columns=["prediction"])
        submit_df["id"] = [i for i in range(len(submit_df))]
        submit_df = submit_df[["id", "prediction"]]

        # Get save path
        submit_path = os.path.join(self.submit_folder_path, self.name + ".csv")

        # Save prediction
        submit_df.to_csv(submit_path, index=False)

        return

    def save_train_valid(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        """
        Saves train and valid results as dataframes
        This can be later used to predict the ensemble train loss and valid loss

        Parameters:
            train_df(pd.DataFrame): Dataframe containing all train data and predicted(y_hat) data
            valid_df(pd.DataFrame): Dataframe containing all valid data and predicted(y_hat) data
        """

        # Create path
        train_path = os.path.join(self.train_folder_path, self.name + "_train.csv")
        valid_path = os.path.join(self.valid_folder_path, self.name + "_valid.csv")

        # Save dataframe as csv
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)

        return


def setup() -> tuple[dict, dict]:
    """
    Setups settings and returns setting/unprocessed

    Returns:
        data(dict): Dictionary containing the unprocessed data dataframes
        settings(dict): Dictionary containing the settings
    """

    # Changes directory to parent directory
    os.chdir("..")

    # Saves project folder path
    folder_path = os.getcwd()

    print("Getting General Settings...")

    # Import settings
    general_settings = get_general_setting(folder_path)

    print("Loaded General Settings!")
    print()

    print("Setting General Settings...")

    # Set basic settings
    set_basic_settings(general_settings)

    print("Set General Setting!")
    print()

    print("Getting Unprocessed Data...")

    # Import unprocessed data
    data = get_unprocessed_data(folder_path, general_settings)

    print("Got Unprocessed Data!")
    print()

    return data, general_settings


def get_save_settings(settings: dict, sid: int) -> SaveSetting:
    """
    Setups settings related to saving files and logs

    Args:
        settings (dict): Dictionary containing the general settings
        sid (int): Unique id for save settings

    Returns:
        save_settings(SaveSetting): Class used to save files(log, model, result)
    """
    # Saves project folder path
    folder_path = os.getcwd()

    print("Getting Save Settings...")

    # Get save settings
    save_settings = SaveSetting(folder_path, settings, sid)

    print("Got Save Settings!")
    print()

    print(f"============== {save_settings.name} ==============")
    print()

    return save_settings


def set_wandb(settings: dict, save_setting: Type[SaveSetting]) -> None:
    if settings["wandb_activate"]:
        print("Set Wandb..")
        wandb.login()
        config = {
            "model_name": settings["model_name"],
            "loss_fn": settings["loss_fn"],
            "optimizer": settings["optimizer"],
            "scheduler": settings["scheduler"],
            "epoch": settings["epoch"],
            "batch_size": settings["batch_size"],
            "num_workers": settings["num_workers"],
            "train_valid_split": settings["train_valid_split"],
        }
        config.update(settings["adam"])
        config.update(settings[settings["model_name"].lower()])
        wandb.init(project="dkt", config=config, reinit=True)
        wandb.run.name = settings["model_name"] + f"_{save_setting.name}"
        wandb.run.save()
        print("Done")
        print()
