"""Implementation of the NT-Xent loss function from [Chen et al. 2020](https://arxiv.org/abs/2002.05709)."""
import torch
from torch import Tensor, nn


class NTXEntLoss(nn.Module):
    """A PyTorch implementation of the NT-Xent loss function for contrastive learning.

    Args:
        temperature (float, optional): Parameter to use for scaling the cosine similarity matrix.
        Defaults to 1.0.

    Inputs:
        - batch_embeddings (Tensor): A tensor of shape `(batch_size, embedding_dim)`
        containing the embeddings for the original data.
        - batch_augmented_embeddings (Tensor): A tensor of shape `(batch_size, embedding_dim)`
        containing the embeddings for the augmented data.

    Returns:
        Tensor: A scalar tensor containing the computed loss value.

    See Also:
        - [Chen et al. 2020 SimCLR paper](https://arxiv.org/abs/2002.05709).

    Example:
        >>> ntx_loss = NTXEntLoss(temperature=0.5)
        >>> embeddings = torch.randn(32, 128)
        >>> augmented_embeddings = torch.randn(32, 128)
        >>> loss = ntx_loss(embeddings, augmented_embeddings)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, batch_embeddings: Tensor, batch_augmented_embeddings: Tensor) -> Tensor:
        """Computes the NT-Xent loss."""
        # Input shape: (batch_size, embedding_dim)
        batch_size = batch_embeddings.size(0)
        embeddings = torch.cat([batch_embeddings, batch_augmented_embeddings], dim=0)

        # Compute the cosine similarity matrix
        sim_matrix = self.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        sim_matrix /= self.temperature

        # Create the negative mask
        negative_mask = (
            ~(torch.eye(batch_size, dtype=bool, device=batch_embeddings.device))  # type: ignore
        ).repeat(2, 2)

        # Compute the loss
        exp_sim_matrix = torch.exp(sim_matrix) * negative_mask.float()
        sum_exp_sim_matrix = torch.sum(exp_sim_matrix, dim=-1)

        pos_indices = torch.arange(batch_size, device=batch_embeddings.device)
        positive_sim = sim_matrix[pos_indices, batch_size + pos_indices]
        negative_sim_sum = sum_exp_sim_matrix[:batch_size]
        loss = -torch.log(torch.exp(positive_sim) / negative_sim_sum).mean()

        return loss
