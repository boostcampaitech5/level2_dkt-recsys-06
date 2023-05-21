from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index


class LightGCN(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    :class:`~torch_geometric.nn.models.LightGCN` learns embeddings by linearly
    propagating them on the underlying graph, and uses the weighted sum of the
    embeddings learned at all layers as the final embedding

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,

    where each layer's embedding is computed as

    .. math::
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

    Two prediction heads and training objectives are provided:
    **link prediction** (via
    :meth:`~torch_geometric.nn.models.LightGCN.link_pred_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.predict_link`) and
    **recommendation** (via
    :meth:`~torch_geometric.nn.models.LightGCN.recommendation_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.recommend`).

    .. note::

        Embeddings are propagated according to the graph connectivity specified
        by :obj:`edge_index` while rankings or link probabilities are computed
        according to the edges specified by :obj:`edge_label_index`.

    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        alpha (float or torch.Tensor, optional): The scalar or vector
            specifying the re-weighting coefficients for aggregating the final
            embedding. If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`~torch_geometric.nn.conv.LGConv` layers.
    """

    def __init__(
        self,
        data: dict,
        settings: dict,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = data["num_nodes"]
        self.embedding_dim = settings["lgcn"]["embedding_dim"]
        self.num_layers = settings["lgcn"]["num_layers"]
        alpha = settings["lgcn"]["alpha"]

        if alpha is None:
            alpha = 1.0 / (self.num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == self.num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (self.num_layers + 1))
        self.register_buffer("alpha", alpha)

        self.embedding = Embedding(self.num_nodes, self.embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
        """
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)

    def predict_link(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
    ) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.

        Args:
            prob (bool, optional): Whether probabilities should be returned.
                (default: :obj:`False`)
        """
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
    ) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.

        Args:
            src_index (torch.Tensor, optional): Node indices for which
                recommendations should be generated.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            dst_index (torch.Tensor, optional): Node indices which represent
                the possible recommendation choices.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            k (int, optional): Number of recommendations. (default: :obj:`1`)
        """
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor, **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (torch.Tensor): The predictions.
            edge_label (torch.Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.num_nodes}, "
            f"{self.embedding_dim}, num_layers={self.num_layers})"
        )
