# High Temperature MaxEnt RL Encourages Agents to Leaern More Similar Policies

This repository explores the generation of more similar behaviors using MaxEnt RL Policies. Our work is described within this paper X.

# Installation

## Base Environemt

To create and activate a Python environment for the projeect, run the following commands:

```bash
conda create --name similar-behavior python=3.12 pip
conda activate similar-behavior
pip install -r requirements.txt
```

## CleanRL SAC Experiments

To recreate our CleanRL experiments, please run the following commands:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git
pip install -r requirements/requirements.txt
```

Then, you can replicate our training runs within a given environment by running the associated slurm file for each of the following environments matching the setting below: 

| Environment | Number of Seeds    | Alpha Values Considered   |
| :---:   | :---: | :---: |
| Hopper-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 |
| HalfCheetah-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5   |
| Walker2d-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5  |
| Ant-v4 | 10   | 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2  |
| Humanoid-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5  |


To run the slum file, for example for Hopper-v4, run the following command to start your slurm training

```bash
bash humanoid-v4.sh
```