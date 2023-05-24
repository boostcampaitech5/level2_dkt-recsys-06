import numpy as np

from src.data import (
    process_data,
    data_split,
    create_datasets,
    create_dataloader,
)
from src.model import create_model
from src.run import run_model
from src.setup import setup, get_save_settings, set_wandb


def main() -> None:
    # Get settings and raw data from files
    data, settings = setup()

    # Process raw data
    process_data(data, settings)

    # Get how many folds to train and test
    num_kfolds = settings["num_kfolds"]

    # Split data into num_kfolds
    train_acc = []
    train_auc = []
    val_acc = []
    val_auc = []
    data_split(data, settings, num_kfolds)
    for k in range(num_kfolds):
        # Get save settings from settings and save id
        save_settings = get_save_settings(settings, k)

        # Set wandb with save settings
        set_wandb(settings, save_settings)

        # Load datasets from k-th fold validation
        dataset = create_datasets(data, settings, k)

        # Create dataloader
        dataloader = create_dataloader(dataset, settings)

        # Create model
        model = create_model(data, settings)

        # Run model
        predicted_data, results = run_model(dataloader, settings, model, save_settings)

        # Save predicted data as csv
        save_settings.save_submit(predicted_data)

        # Close log if opened
        save_settings.close_log()

        train_acc.append(results["train_acc"])
        train_auc.append(results["train_auc"])
        val_acc.append(results["valid_acc"])
        val_auc.append(results["valid_auc"])

    avg_train_acc = np.array(train_acc).mean()
    avg_train_auc = np.array(train_auc).mean()
    avg_val_acc = np.array(val_acc).mean()
    avg_val_auc = np.array(val_auc).mean()

    print()
    print(
        f"Avg results:\tTrain AUC: {avg_train_auc}\tTrain ACC: {avg_train_acc}\n"
        + f"Avg results:\tValid AUC: {avg_val_auc}\tValid ACC: {avg_val_acc}\n"
    )

    return


if __name__ == "__main__":
    main()
