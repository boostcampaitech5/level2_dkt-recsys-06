import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import sigmoid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
import wandb


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss


def run_model(dataloader: dict, settings: dict, model, save_settings):
    """
    Runs model through train, valid, and submit.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        settings(dict): Dictionary containing the settings.
        model(nn.Module): Model used to train
    """

    # Set loss function
    if settings["loss_fn"].lower() == "rmse":
        loss_fn = RMSELoss()
    elif settings["loss_fn"].lower() == "mse":
        loss_fn = MSELoss()
    elif settings["loss_fn"].lower() == "bcewll":
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Set optimizer
    if settings["optimizer"].lower() == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=settings["adam"]["learn_rate"],
            weight_decay=settings["adam"]["weight_decay"],
        )

        optimizer.zero_grad()

    if settings["scheduler"].lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=settings["plateau"]["patience"],
            factor=settings["plateau"]["factor"],
            mode=settings["plateau"]["mode"],
            verbose=settings["plateau"]["verbose"],
        )

    print("Training Model...")
    print()

    best_auc = -1

    # Set epoch for training
    for epoch in range(settings["epoch"]):
        # Change model state to train
        model.train()

        # Get average loss while training
        if not settings["is_graph_model"]:
            train_auc, train_acc = train_model(
                dataloader, model, loss_fn, optimizer, scheduler, settings
            )
        else:
            train_auc, train_acc = train_graph_model(
                dataloader["train"], model, optimizer
            )

        # Change model state to evaluation
        model.eval()

        # Get average loss using validation set
        if not settings["is_graph_model"]:
            valid_auc, valid_acc = validate_model(dataloader, model, loss_fn, settings)
        else:
            valid_auc, valid_acc = validate_graph_model(dataloader["valid"], model)

        if valid_auc > best_auc:
            best_auc = valid_auc

        scheduler.step(best_auc)

        # Print average loss of train/valid set
        print(
            f"Epoch: {epoch + 1}\nTrain acc: {train_acc}\tTrain auc: {train_auc}\nValid acc: {valid_acc}\t Valid auc: {valid_auc}\n"
        )

        save_settings.append_log(
            f"Epoch: {epoch + 1}\nTrain acc: {train_acc}\tTrain auc: {train_auc}\nValid acc: {valid_acc}\t Valid auc: {valid_auc}"
        )
        if settings["wandb_activate"]:
            wandb.log(
                dict(
                    train_acc_epoch=train_acc,
                    train_auc_epoch=train_auc,
                    valid_acc_epoch=valid_acc,
                    valid_auc_epoch=valid_auc,
                )
            )

    print()

    print("Trained Model!")
    print()

    print("Getting Final Results...")

    # Get final results
    if not settings["is_graph_model"]:
        train_df, train_final_auc, train_final_acc = get_df_result(
            dataloader["train"], model, loss_fn, settings
        )
        valid_df, valid_final_auc, valid_final_acc = get_df_result(
            dataloader["valid"], model, loss_fn, settings
        )
    else:
        train_df, train_final_auc, train_final_acc = graph_get_df_result(
            dataloader["train"], model
        )
        valid_df, valid_final_auc, valid_final_acc = graph_get_df_result(
            dataloader["valid"], model
        )

    save_settings.save_train_valid(train_df, valid_df)

    print(
        f"Final results:\tTrain AUC: {train_final_auc}\tTrain ACC: {train_final_acc}\n"
        + f"Final results:\tValid AUC: {valid_final_auc}\tValid ACC: {valid_final_acc}\n"
    )

    print("Got Final Results!")
    print()

    print("Saving Model/State Dict...")

    # Save model and state_dict, loss, settings
    save_settings.save_model(model)
    save_settings.save_statedict(
        model,
        train_final_auc,
        train_final_acc,
        valid_final_auc,
        valid_final_acc,
        settings,
    )

    print("Saved Model/State Dict!")
    print()

    print("Predicting Results...")

    # Get predicted data for submission
    if not settings["is_graph_model"]:
        predict_data = test_model(dataloader, model, settings)
    else:
        predict_data = test_graph_model(dataloader["test"], model)
    print("Predicted Results!")
    print()

    return predict_data


def train_model(
    dataloader: dict, model, loss_fn, optimizer, scheduler, settings
) -> float:
    """
    Trains model using train data.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train
        loss_fn: Used to find the loss between two tensors
        optimizer: Used to optimize parameters
    """

    total_preds = []
    total_targets = []
    losses = []

    for data in dataloader["train"]:
        # Data to device
        data = {k: v.to(settings["device"]) for k, v in data.items()}

        # Split data to input and output
        x = data
        y = data[settings["predict_column"]]

        # Get predicted output with input
        y_hat = model(x)

        # Get loss using predicted output
        loss = loss_fn(y_hat, y.float())

        loss = loss[:, -1]
        loss = torch.mean(loss)

        # Computes the gradient of current parameters
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 10)

        # Optimize parameters
        optimizer.step()

        # Set the gradients of all optimized parameters to zero
        optimizer.zero_grad()

        y_hat = sigmoid(y_hat[:, -1])
        y = y[:, -1]

        total_preds.append(y_hat.detach())
        total_targets.append(y.detach())
        losses.append(loss.detach())

    total_targets = torch.concat(total_targets).cpu().numpy()
    total_preds = torch.concat(total_preds).cpu().numpy()

    auc = roc_auc_score(y_true=total_targets, y_score=total_preds)
    acc = accuracy_score(
        y_true=total_targets, y_pred=np.where(total_preds >= 0.5, 1, 0)
    )

    return auc, acc


def validate_model(dataloader: dict, model, loss_fn, settings) -> float:
    """
    Uses valid dataloader to get loss of model.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train
        loss_fn: Used to find the loss between two tensors
    """

    total_preds = []
    total_targets = []

    # No learning from validation data
    with torch.no_grad():
        for data in dataloader["valid"]:
            # Data to device
            data = {k: v.to(settings["device"]) for k, v in data.items()}

            # Split data to input and output
            x = data
            y = data[settings["predict_column"]]

            # Get predicted output with input
            y_hat = model(x)

            y_hat = sigmoid(y_hat[:, -1])
            y = y[:, -1]

            total_preds.append(y_hat.detach())
            total_targets.append(y.detach())

    total_targets = torch.concat(total_targets).cpu().numpy()
    total_preds = torch.concat(total_preds).cpu().numpy()

    auc = roc_auc_score(y_true=total_targets, y_score=total_preds)
    acc = accuracy_score(
        y_true=total_targets, y_pred=np.where(total_preds >= 0.5, 1, 0)
    )

    return auc, acc


def test_model(dataloader: dict, model, settings) -> list:
    """
    Use test data to get prediction for submission.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train

    Returns:
        predicted_list(list): Predicted results from test dataset.
    """
    # Predicted values in order
    predicted_list = list()

    with torch.no_grad():
        for data in dataloader["test"]:
            # Data to device
            data = {k: v.to(settings["device"]) for k, v in data.items()}

            # Get input data
            x = data

            # Get predicted output with input
            y_hat = model(x)

            y_hat = sigmoid(y_hat[:, -1])
            y_hat = y_hat.cpu().detach().numpy()

            # Add predicted output to list
            predicted_list += list(y_hat)

    return predicted_list


def train_graph_model(
    train_data: dict, model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple:
    pred = model(train_data["edge"])
    loss = model.link_pred_loss(pred=pred, edge_label=train_data["label"])

    prob = model.predict_link(edge_index=train_data["edge"], prob=True)
    prob = prob.detach().cpu().numpy()

    label = train_data["label"].cpu().numpy()
    acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
    auc = roc_auc_score(y_true=label, y_score=prob)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return auc, acc


def validate_graph_model(valid_data: dict, model: nn.Module) -> tuple:
    with torch.no_grad():
        prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        prob = prob.detach().cpu().numpy()

        label = valid_data["label"].detach().cpu().numpy()
        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)

    return auc, acc


def test_graph_model(test_data: dict, model: nn.Module) -> list:
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=test_data["edge"], prob=True)

    pred = pred.detach().cpu().numpy()
    return list(pred)


def get_df_result(dataloader, model, loss_fn, settings):
    """
    Gets prediction to get as output csv

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train
        loss_fn: Used to find the loss between two tensors
        optimizer: Used to optimize parameters
    """

    total_preds = []
    total_targets = []

    # Create dataframe to save

    with torch.no_grad():
        for data in dataloader:
            # Data to device
            data = {k: v.to(settings["device"]) for k, v in data.items()}

            # Split data to input and output
            x = data
            y = data[settings["predict_column"]]

            # Get predicted output with input
            y_hat = model(x)

            y_hat = sigmoid(y_hat[:, -1])
            y = y[:, -1]

            total_preds.append(y_hat)
            total_targets.append(y)

    total_targets = torch.concat(total_targets).cpu().numpy()
    total_preds = torch.concat(total_preds).cpu().numpy()

    auc = roc_auc_score(y_true=total_targets, y_score=total_preds)
    acc = accuracy_score(
        y_true=total_targets, y_pred=np.where(total_preds >= 0.5, 1, 0)
    )

    save_df = pd.Series(total_preds)

    return save_df, auc, acc


def graph_get_df_result(dataloader: dict, model: nn.Module) -> tuple:
    with torch.no_grad():
        prob = model.predict_link(edge_index=dataloader["edge"], prob=True)
        prob = prob.detach().cpu().numpy()

        label = dataloader["label"].detach().cpu().numpy()
        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)

    save_df = pd.Series(prob)

    return save_df, auc, acc
