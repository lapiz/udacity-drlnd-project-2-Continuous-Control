# Project 2 : Continuous Control

This repository contains an implementation of project 2 for [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details

TBD

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

TBD

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
- scores.py
  - helper code for score data
