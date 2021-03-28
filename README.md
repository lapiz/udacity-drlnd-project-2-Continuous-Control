# Project 2 : Continuous Control

This repository contains an implementation of project 2 for [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details

In this project, I trained agents with double-jointed arm, According to project introduciton, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Project have two options, Single agent vs 20 agents. I challenge to 20 agents. My all agents get an average score of +30 over 100 consecutive episodes.

## Getting Started

### Python environment

- If you run this project on your own environment, you install some packages.
  - Python == 3.6
  - pytorch == 0.4
  - mlagents == 0.4 (Unity ML Agents)
- Or you can run this project on jupyter notebook.

### Dependencies

To set up your python environment (with conda) to run code in the project, follow the intstruction below.

- Create and activate a new envirionment with Python 3.6

```bash
conda create --name project2 python=3.6
conda activate project2
```

- Clone my project repository and install requirements.txt

```bash
git clone https://github.com/lapiz/udacity-drlnd-project-2-Continuous-Control.git
cd udacity-drlnd-project-2-Continuous-Control
pip install -r requirements.txt
```

### Downloading the Unity environment

Different versions of the Unity environment are required on different operational systems.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
  
## Instructions

- Command lines
  - If train yourself, run it. you can skip hparam filename, it found 'default.json' file.

```bash
python train.py hparam.json
```

- Notebook
  - If train, open [Continuous_Control.ipynb](Continuous_Control.ipynb)
  - If run with already trained data, open [run_trained.ipynb](run_trained.ipynb)

## Files

- README.md
  - This file
- requirements.txt
  - python environment requirements packages
  - Use pip with -r options
- Continuous_Control.ipynb
  - Main notebook file.
  - Based on udacity project skelecton notebook
  - I implemented my agent and some helper classes.
- run_trained.ipynb
  - Run with trained data sample
- Report.ipynb
  - My Project report.
  - Include these things
    - Learning Algorithm
      - Hyperpameters
      - Model architechures
    - Plot of Rewards
- reacher_result_actor.pth
  - trained model weights for actor
- reacher_result_critic.pth
  - trained model weights for critic
- scores.py
  - helper code for score data
- train.py
  - Train ddpg agents.
  - Based on udacity DRLND ddqg-bipdel sample project
  - Run ommand line with hparam json arguments for test
- ddpg_agent.py
  - DDPG Agent implementation with PlayBuffer
  - Based on udacity DRLND ddqg-bipdel sample project
  - Modifiied for sharing models with 20 agents.
- model.py
  - Model described by hidden layers (Actor and Critic)
