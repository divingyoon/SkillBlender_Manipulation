<h1 align="center">
  <b>SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending</b>
</h1>

This is the official code release of our paper [SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending](https://arxiv.org/abs/2506.09366). This repository contains code and ckpts of our framework **SkillBlender** and benchmark **SkillBench**.

**[`[Paper]`](https://arxiv.org/abs/2506.09366) [`[Project Page]`](https://usc-gvl.github.io/SkillBlender-web/) [`[Checkpoints (Google Drive)]`](https://drive.google.com/drive/folders/104W9oBdHxkOKFUZNV60jQo4sugaM0rit?usp=drive_link) [`[Checkpoints (Hugging Face)]`](https://huggingface.co/yxK/SkillBlender_ckpt_release)**

<div align=center>
    <img src="docs/img/teaser.jpg" width=100%>
</div>

## Installation

Our code is easy to install and run. Please refer to [1.INSTALLATION.md](./docs/1.INSTALLATION.md) for detailed instructions.

## Training, playing, and evaluation

Please refer to [2.RUNNING.md](./docs/2.RUNNING.md) for how to train, play, and evaluate the skills and tasks of SkillBlender in our SkillBench.
We also include some troubleshooting tips in this document.

## Changing observations

We support both state-based and vision-based observations. The default observation is state-based, which includes joint positions, velocities, etc. You can also use ego-centric vision observations for high-level tasks.
Please refer to [3.OBSERVATION.md](./docs/3.OBSERVATION.md) for more details on how to change the observation types and structures.

## Changing humanoid

We support three distinct humanoid robots: H1, G1, and H1-2. Each robot has its own set of skills and tasks. However, you can also easily change the humanoid robot in the codebase to your own robot by modifying the config files and the robot class.
For more details, please refer to [4.HUMANOID.md](./docs/4.HUMANOID.md).

## Release status

- [x] Release full code of our framework **SkillBlender** and benchmark **SkillBench**, including simulation environments, rewards, training & playing scripts, etc.
- [x] Release pretrained ckpts for H1 primitive skills and loco-manipulation tasks.
- [ ] Release pretrained ckpts for G1 and H1-2 primitive skills and loco-manipulation tasks.
- [ ] Release sim2sim and sim2real code for primitive skill deployment.
- More to come... (Feel free to open issues and PRs!)

**Please stay tuned for any updates of this repository!**

## Acknowledgement

This project is based on [legged_gym](https://github.com/leggedrobotics/legged_gym), [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [humanoid-gym](https://github.com/roboterax/humanoid-gym), [humanplus](https://github.com/MarkFzp/humanplus), and [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym). We thank the authors for their contributions.

## Citation

If you find this work helpful, please consider citing:

```
@article{kuang2025skillblender,
  title={SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending},
  author={Kuang, Yuxuan and Geng, Haoran and Elhafsi, Amine and Do, Tan-Dzung and Abbeel, Pieter and Malik, Jitendra and Pavone, Marco and Wang, Yue},
  journal={arXiv preprint arXiv:2506.09366},
  year={2025}
}
```