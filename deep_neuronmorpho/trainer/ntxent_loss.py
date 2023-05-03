"""Implementation of the NT-Xent loss function from [Chen et al. 2020](https://arxiv.org/abs/2002.05709)."""
import torch
from torch import Tensor, nn


class NTXEntLoss(nn.Module):
    """A PyTorch implementation of the NT-Xent loss function for contrastive learning.

    Args:
        temperature (float, optional): Parameter to use for scaling the cosine similarity matrix.
        Defaults to 1.0.

    Inputs:
        - embed_a (Tensor): A tensor of shape `(batch_size, embedding_dim)`
        containing the embeddings for the original data.
        - embed_b (Tensor): A tensor of shape `(batch_size, embedding_dim)`
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

    def forward(self, embed_a: Tensor, embed_b: Tensor) -> Tensor:
        """Computes the NT-Xent loss."""
        # Input shape: (batch_size, embedding_dim)
        embeddings = torch.cat([embed_a, embed_b], dim=0)
        n_samples = len(embeddings)
        # Compute cosine similarity matrix
        sim = self.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        sim_exp = torch.exp(sim / self.temperature)
        # Remove diagonal elements (self-similarity)
        mask = ~torch.eye(n_samples, device=sim_exp.device).bool()
        neg = sim_exp.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # Positive similarity
        pos = torch.exp(torch.sum(embed_a * embed_b, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()

        return loss
