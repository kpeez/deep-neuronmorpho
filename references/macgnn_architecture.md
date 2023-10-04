# MACGNN architecture

## Overview

The Morphology-aware contrastive GNN model (MACGNN) model from [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206) is a graph neural network that uses contrastive learning to learn graph-level embeddings in an unsupervised manner using the NT-Xent loss.

> Note: We are not implementing the binary hashing, just the contrastive learning phase.

## Model Architecture

MACGNN is based on a 5-layer GIN model (input + 4 GIN layers). This model updates each node's representations by using sum aggregation for a node and its neighbors. It then uses a MLP to update the node's representation:
$$
h_{v}^{(k)} = \text{MLP}^{k}\bigg( \big( 1 + \epsilon \big) \cdot h_{v}^{(k-1)} \sum\limits_{u \in N(v)} h^{(k-1)} \bigg)
$$

- MLP is a 2-layer (1 hidden layer) MLP with BatchNorm -> ReLU.

    ```markdown
    MLP(
    (mlp): Sequential(
        (0): Linear(in_features=in_dim, out_features=in_dim, bias=True)
        (1): BatchNorm1d(in_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=in_dim, out_features=in_dim, bias=True)
    )
    )
    ```

- Each GIN layer also uses dropout with a probability of 0.5.
- MACGNN uses two different GIN networks of identical architecture, but with different node features.
  - one GIN network uses the geometric node features (x, y, z) position of neuron
  - the other GIN network uses the topological node features (euclidean distance, path distance from soma, and angle and branch statistics).

### Forward pass

- use a single linear prediction layer to get the node embeddings
- apply a graph pooling layer (MAX or SUM) to get the graph embedding
- collect the graph-level pooled representation from each layer and aggregate them
  - aggregation is done either via CONCAT or SUM
- Finally, this concatenated or summed representation is combined via a weighted sum (with learnable weight parameter) and passed through a single linear layer to get the final graph embedding
- For training, we get embeddings for the batch as well as the augmented batch and pass them through the NT-Xent loss.
