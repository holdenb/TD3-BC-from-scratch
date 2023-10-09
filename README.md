# Re-implementation of "A Minimalist Approach to Offline Reinforcement Learning"

[Link to original GitHub repo](https://github.com/sfujim/TD3_BC/tree/main#a-minimalist-approach-to-offline-reinforcement-learning)

TD3+BC is a simple approach to offline RL where only two changes are made to TD3: (1) a weighted behavior cloning loss is added to the policy update and (2) the states are normalized. Unlike competing methods there are no changes to architecture or underlying hyperparameters (Fujimoto & Gu, 2021). The paper can be found [here](https://arxiv.org/abs/2106.06860).

## Details

Paper results were collected with [MuJoCo 1.50](http://www.mujoco.org/) (and [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.

### BibTex

Please cite the primary authors if any code is used or referenced.

```bibtex

@inproceedings{fujimoto2021minimalist,
 title={A Minimalist Approach to Offline Reinforcement Learning},
 author={Scott Fujimoto and Shixiang Shane Gu},
 booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
 year={2021},
}

```

---
