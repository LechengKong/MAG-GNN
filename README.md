# MAG-GNN
The official implementation of [MAG-GNN: Reinforcement Learning Boosted Graph
Neural Network](https://arxiv.org/abs/2310.19142), accepted by NeurIPS 2023.

## Requirements

Install the required packages by

`conda env create -f environment.yml`

## Experiments

The experiments can be ran by:

`python rl_main.py`

To specify a dataset:

`python rl_main.py --data_config_path ./configs/dataset_configs/zinc.yaml`

All dataset configs resides in ./configs/dataset_configs.

To specify a experiment:

`python rl_main.py --exp_config_path ./configs/dataset_configs/ord.yaml`

All experiment configs resides in ./configs/exp_configs.

`ord.yaml` for ORD paradigm, `simul.yaml` for SIMUL paradigm, `pre.yaml` for PRE paradigm.

For example, to ran ZINC experiments with SIMUL paradigm,

`python rl_main.py --data_config_path ./configs/dataset_configs/zinc.yaml --exp_config_path ./configs/dataset_configs/ord.yaml`

Parameters can be override either by a override yaml file

`python rl_main.py --override override.yaml`

Or by a space separated command-line arguments.

`python rl_main.py --data_config_path ./configs/dataset_configs/zinc.yaml num_layers 5 num_epochs 100 lr 0.0001`


### PRE paradigm experiments

First train the RL agent using any SIMUL/ORD paradigm, for example

`python rl_main.py --data_config_path ./configs/dataset_configs/syn_count.yaml --exp_config_path ./configs/dataset_configs/ord.yaml`

After training, the RL agent checkpoints will be saved at 
`./saved_exp/{datetime}/{exp_type}_{dataset_name}/{hash}/checkpoints/*`

The command for the RL agent will be saved at `./saved_exp/{datetime}/command`

Pick a checkpoint and override the `mover_load` parameter in the yaml file, either directly in `pre.yaml`, or by a override yaml file

```yaml
mover_load:
  - {command path}
  - {checkpoint path}
```

### Dataset specific arguments

For QM dataset: override `data_label` argument to specify target labels from the list:
```python
["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv", ]
```

For syn_count dataset: override `data_label` argument to specify `data_label`+3-cycle. `data_label` in [0, 3]
