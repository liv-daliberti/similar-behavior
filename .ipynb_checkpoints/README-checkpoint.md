# High Temperature MaxEnt RL Encourages Agents to Leaern More Similar Policies

This repository explores the generation of more similar behaviors using MaxEnt RL Policies. Our work is described within this paper X.

# Installation

## Base Environemt

To create and activate a Python environment for the projeect, run the following commands:

```bash
conda create --name similar-behavior python=3.10 pip
conda activate similar-behavior
pip install -r requirements.txt
```

## CleanRL SAC Experiments

To recreate our CleanRL experiments, please run the following commands:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
pip install -r requirements/requirements.txt
```

Then, you can replicate the data used within our experiments by training runs within a given environment by running the associated slurm file for each of the following environments matching the setting below: 

| Environment | Number of Seeds    | Alpha Values Considered   | Steps of Training | Save Frequency (Steps)|
| :---:   | :---: | :---: | :---: |:---: |
| Hopper-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 | 3M | 50K |
| HalfCheetah-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5   | 3M | 50K |
| Walker2d-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5  | 8M | 50K |
| Ant-v4 | 10   | 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2  | 8M| 50K |
| Humanoid-v4 | 10   | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5  | 10M | 50K |

We consider experimental results for several common maxEnt RL methods, including SAC, MEOW, PCL, and NAF. The defaults are set for SAC.

You must add the following segment to the bottom of your cleanrl script in order to ensure saving:

```
# Inside your training loop, after processing the batch (e.g., after logging)
if global_step % 50000 == 0:
    checkpoint = {
        'global_step': global_step,
        'actor_state_dict': actor.state_dict(),
        'qf1_state_dict': qf1.state_dict(),
        'qf2_state_dict': qf2.state_dict(),
        'qf1_target_state_dict': qf1_target.state_dict(),
        'qf2_target_state_dict': qf2_target.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'q_optimizer_state_dict': q_optimizer.state_dict(),
        'args': vars(args)  # Save current hyperparameters for reproducibility.
    }
    checkpoint_path = f"checkpoints/{run_name}_step{global_step}.pth"
    # Ensure the checkpoints directory exists.
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
```


To run the slum file, for example for Hopper-v4, run the following command to start your slurm training

```bash
bash humanoid-v4.sh
```