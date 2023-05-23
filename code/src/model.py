from .model_folder.model_lstm import LongShortTermMemory
from .model_folder.model_lstmattn import LongShortTermMemoryAttention
from .model_folder.model_bert import BidirectionalEncoderRepresentationsfromTransformers
from torch_geometric.nn.models import LightGCN


def create_model(data: dict, settings: dict):
    """
    Creates model using settings.

    Parameters:
        settings(dict): Dictionary containing the settings.

    Returns:
        model(nn.Module): Model based on settings.
    """

    print("Creating Model...")

    # Get model
    if settings["model_name"].lower() == "mlp":
        model = MultiLayerPerceptronClass(settings, input_dim=settings["column_num"])
    elif settings["model_name"].lower() == "lstm":
        model = LongShortTermMemory(settings)
    elif settings["model_name"].lower() == "lstm_attn":
        model = LongShortTermMemoryAttention(settings)
    elif settings["model_name"].lower() == "bert":
        model = BidirectionalEncoderRepresentationsfromTransformers(settings)
    elif settings["model_name"].lower() == "lgcn":
        model = LightGCN(
            num_nodes=data["num_nodes"],
            embedding_dim=settings["lgcn"]["embedding_dim"],
            num_layers=settings["lgcn"]["num_layers"],
            alpha=settings["lgcn"]["alpha"],
        )
    else:
        print("No model found ending program")

    print("Created Model!")
    print()

    # Return model with device
    return model.to(settings["device"])
