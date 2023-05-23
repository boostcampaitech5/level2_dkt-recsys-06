from src.setup import setup, get_save_settings, set_wandb
from src.data import (
    process_data,
    data_split,
    create_datasets,
    create_dataloader,
)
from src.model import create_model
from src.run import run_model

import optuna
import os, sys, json

save_dir = os.path.abspath("../best_params/")
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


def objective(trial):
    model_name = default_settings["model_name"]
    # params = default_settings[model_name]

    # randomize hyperparameters

    # 1. general hyperparameters
    # default_settings['loss_fn'] = trial.suggest_categorical("loss_fn", ["BCEWLL", "mse", "rmse"])
    default_settings["epoch"] = trial.suggest_int("epochs", 1, 2, step=1, log=False)
    default_settings["batch_size"] = trial.suggest_int(
        "batchsize", 8, 64, step=4, log=False
    )
    if "max_seq_len" in default_settings.keys():
        default_settings["max_seq_len"] = trial.suggest_int(
            "max_seq_len", 4, 40, log=False
        )
    # default_settings['optimizer'] = trial.suggest_categorical("optimizer", ["adam", "SGD", "RMSprop", "Adadelta"])

    # 2. special hyperparameters
    if model_name == "lgcn":
        default_settings["alpha"] = trial.suggest_float("drop_out", 0.0, 1.0, log=False)
    elif model_name == "lstm_attn":
        default_settings["drop_out"] = trial.suggest_float(
            "drop_out", 0.0, 1.0, log=False
        )

    # Create model
    model = create_model(data, default_settings, silence=True)
    # Run model
    return run_model(
        dataloader, default_settings, model, save_settings, tune=True, silence=True
    )


# Get settings and raw data from files (os.getcwd changes to entire folder)
data, settings = setup(silence=True)

# Process raw data
process_data(data, settings, silence=True)

# Split data
data_split(data, settings, 1, silence=True)

# Get save settings
save_settings = get_save_settings(settings, 0)

# Set wandb
set_wandb(settings, save_settings)

# Load datasets
dataset = create_datasets(data, settings, 0, silence=True)

# Create dataloader
dataloader, default_settings = create_dataloader(
    dataset, settings, tune=True, silence=True
)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2)
with open(f'{save_dir}/{default_settings["model_name"]}.json', "w") as f:
    json.dump(study.best_params, f, indent="\t")
