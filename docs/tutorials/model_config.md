# Model configuration file format

For training models, a model configuration file is required. The configuration file is a YAML file with the following structure:

```yaml
dirs:
  graph_data: "path_to_graph_data_dir"
  expt_results: "path_to_expt_results_dir"

datasets:
  contra_train: "contrastive_training_dataset"
  eval_train: "evaluation_training_dataset"
  eval_test: "evaluation_test_dataset"

model:
  name: "model_name"
  num_gnn_layers: 
  num_mlp_layers: 
  hidden_dim: # dimensionality of the hidden layers in GNN
  output_dim:  # hidden_dim * num_gnn_layers
  dropout_prob: # dropout probability
  attrs_streams:
    geometric: # indices of geometric attributes (x, y, z)
    topological: # indices of topological attributes:  euc_dist, path_dist, angles/branches
  use_edge_weight: True # attention
  learn_eps: False # learnable eps in GINConv
  graph_pooling_type: "max" # "sum", "mean", "max"
  neighbor_aggregation: "sum" # "sum", "mean", "max"
  gnn_layers_aggregation: "cat" # ["sum", "mean", "max", "wsum", "cat"]
  stream_aggregation: "wsum" # ["sum", "mean", "max", "wsum", "cat"]

# Training configuration
training:
  batch_size: 64
  contra_loss_temp: 0.2
  dual_aug_loss: False
  eval_interval: 1
  max_epochs: 400
  patience: 60
  optimizer: "adam"
  lr_init: 0.01
  lr_scheduler: "step" # ["cosine", "step"]
  lr_decay_steps: 20
  lr_decay_rate: 0.6
  random_seed: 42 # optional

# Augmentation info
augmentation:
  order: [perturb, rotate, drop_branches]
  params:
    perturb:
      prop: 0.5
      std_noise: 2.0
    rotate: {} # no args needed
    drop_branches:
      prop: 0.02
      deg_power: 1.0
      path_dist_idx: 4
```
