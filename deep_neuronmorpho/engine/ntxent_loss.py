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
        ```
        ntx_loss = NTXEntLoss(temperature=0.5)
        embeddings = torch.randn(32, 128)
        augmented_embeddings = torch.randn(32, 128)
        loss = ntx_loss(embeddings, augmented_embeddings)
        ```
    """

    def __init__(self, temperature: float | None = None) -> None:
        super().__init__()
        self.temperature = temperature or 1.0
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.mask: Tensor | None = None

    def forward(self, embed_a: Tensor, embed_b: Tensor) -> Tensor:
        """Computes the NT-Xent loss."""
        embeddings = torch.cat([embed_a, embed_b], dim=0)
        n_samples = len(embeddings)
        # Create mask for diagonal elements if not already cached or if batch size changed
        if self.mask is None or self.mask.shape[0] != n_samples:
            self.mask = ~torch.eye(n_samples, device=embeddings.device).bool()
        # Compute cosine similarity matrix
        sim = self.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        sim_exp = torch.exp(sim / self.temperature)
        # Remove diagonal elements (self-similarity)
        neg = sim_exp.masked_select(self.mask).view(n_samples, -1).sum(dim=-1)
        # Positive pair similarities for each pair, both (i,j) and (j,i)
        pos = torch.diag(sim_exp[: len(embed_a), len(embed_b) :]).repeat(2)
        loss = -torch.log(pos / neg).mean()

        return loss
