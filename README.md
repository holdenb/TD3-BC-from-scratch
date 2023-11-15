# Re-implementation of "A Minimalist Approach to Offline Reinforcement Learning"

[Original TD3 Implementation](https://github.com/sfujim/TD3)

[Original TD3+BC Implementation](https://github.com/sfujim/TD3_BC)

[Original BCQ Implementation](https://github.com/sfujim/BCQ)

TD3+BC is a simple approach to offline RL where only two changes are made to TD3: (1) a weighted behavior cloning loss is added to the policy update and (2) the states are normalized. Unlike competing methods there are no changes to architecture or underlying hyperparameters (Fujimoto & Gu, 2021). The paper can be found [here](https://arxiv.org/abs/2106.06860).

## Details

Original Paper results were collected with [MuJoCo 1.50](http://www.mujoco.org/) (and [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.

The experiments utilize our own rewritten version of TD3+BC utilizing the dependencies described above and utilize the DR4L benchmark of OpenAI gym MuJoCo tasks. All D4RL datasets use the v0 version to replicate experiments and baselines conducted by Fujimoto et al. in the original paper.

## Description

Each simulated environment has five difficulties: Random, Medium, Medium Replay, Medium Expert, and Expert. Each simulation contains a goal, and an agent is scored at the end of an evaluation based on appropriate actions that the agent took to attain the goal.  Our re-implementation of TD3+BC is trained for for 1 million time steps and evaluated every 5000 time steps. Each evaluation consists of 10 episodes of the gym simulation.  The average normalized D4RL score over the final 10 evaluations and 5 seeds is then used to plot the learning curves for each task, where a learning curve consists of the normalized score over the entire duration of training.

Fujimoto et al. notes in *Addressing Function Approximation Error in Actor-Critic Methods* that the original TD3 implementation utilizes a two layer feed-forward neural network of 400 and 300 hidden nodes respectively with rectified linear units (ReLU) between each layer for both the actor and critic, and a final tanh unit following the output of the actor. TD3+BC utilizes 256 hidden nodes between both the actor and critic. Our implementation adds the original 400 and 300 hidden nodes to TD3+BC. Our implementation continues to use a batch size of 256.

## How to Run the Experiments

After installing the python dependencies simply run the provided script

```sh
./run_experiments.sh
```

This should kick off a training run on each experiment listed. The TD3+BC agent is trained on each environment as stated in the description.

This script is directly referenced from the original implementation. This is needed to reproduce results across the environments used in the original testing.

To visualize results, the `TD3_BC_results.ipynb` can be used to plot the normalized DR4L scores at each time step. The script should overlay runs of each environment. The `num_eval_per_scenario` parameter can be modified according to how many runs were produced per scenario i.e. `[hopper-random-v0, hopper-medium-v0, hopper-expert-v0]` would be `3` evaluations for the `hopper` scenario.

## Code References

`main.py` is our own rewritten setup implementation. `fmt_output` is shared in the evaluation of D4RL metrics. The original implementation is referenced for `eval_policy`, however some performance enhancements were added that simplify evaluation. This implementation utilizes a much simpler startup and partitions out the environment seeding and the evaluation.

`replay_buffer.py` is heavily referenced from the original by Fujimoto et al. in the original TD3+BC implementation, however we have stripped out the unnecessary functions that were used for online training. Also some parameters relating to round-robin choosing of random samples from the batch were removed due to that being unnecessary in offline training, since we only get one batch per run.

`TD3_BC.py` utilizes the same actor/critic as the original, however we have increased the hidden layer size to 400 and 300 nodes respectively (up from 256), while keeping the incoming batch size fixed at 256. Our implementation has functioned out a lot of necessary components used in the `train` method (`select_action`, `compute_noise`, etc.). We were also able to simplify the training step to use less memory (hold less tensors in memory at once) and reduce the excess copying of tensors.

`utils.py` references the main repository for the defaults and default hyperparameters. The structure of the code and utility methods are all new for this implementation.

Results can be visualized using the provided notebook `TD3_BC_results.ipynb`. This is our custom script to visualize results produced in the `/results` directory.

## Datasets

As mentioned in the details, the datasets used are [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Offline RL utilizes batch-constrained learning, so on a given run a single batch is acquired from the gym environment and used throughout training and evaluation. The results are evaluated using the D4RL normalized scores and are based on each specific MuJoCu task.

## TODO

- [ ] Run v2 envs and compare against paper v0 baselines
- [ ] Add dockerfile for easy reproduction of the experiments
- [ ] Fix args to allow for model saving & hyperparam tuning
- [ ] Add option for internally benchmarking each evaluation phase (save to npy similar to D4RL scores)
- [ ] Update to use [Gymnasium-Robotics](https://gymnasium.farama.org/environments/mujoco/) and
[Minari](https://github.com/Farama-Foundation/Minari). D4RL is planning to support Minari.
- [ ] Replicate the [D4RL normalized score](https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71) for the Minari envs

### BibTex

Please cite the primary authors if any code is used or referenced.

#### TD3+BC Reference & Paper

```bibtex

@inproceedings{fujimoto2021minimalist,
 title={A Minimalist Approach to Offline Reinforcement Learning},
 author={Scott Fujimoto and Shixiang Shane Gu},
 booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
 year={2021},
}

```

#### TD3 (Online Algorithm) Reference & Paper

```bibtex

@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}
}

```

#### Batch-Constrained Deep Q-Learning (BCQ) References & Papers

```bibtex

@inproceedings{fujimoto2019off,
  title={Off-Policy Deep Reinforcement Learning without Exploration},
  author={Fujimoto, Scott and Meger, David and Precup, Doina},
  booktitle={International Conference on Machine Learning},
  pages={2052--2062},
  year={2019}
}

```

```bibtex

@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti, Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}

```

---
