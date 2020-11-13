# Collaboration and competition using Reinforcement Learning
This project is aimed to train a pair of agents to play tennis. 

We use [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) to train the agents with a help of a common critic.

# Environment details

In this environment, the two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

#### *State and action space*:

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. There are a total of 24 states and 2 actions for each agent.


# Environment Setup

We utilize [Unity Machine Learning Agents](https://github.com/Unity-Technologies/ml-agents) plugin to interact with the environment. 

To set up your python environment to run the code in this repository, follow the instructions below:

1. Create (and activate) a new [Conda](https://docs.anaconda.com/anaconda/install/) environment with Python 3.6.

    ```bash
    conda create --name <env_name> python=3.6
    conda activate <env_name>
    ```


2. Clone the repository, and navigate to the **python/** folder. Then, install several dependencies.

    ```bash
    git clone https://github.com/Akshai/rl_collaboration_and_competition.git
    cd rl_collaboration_and_competition/python
    pip install .
    ```
    
    
3.  Download the environment from one of the links below. You need to only select the environment that matches your operating system:

    Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) <br />
    Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)<br />
    Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)<br />
    Windows (64-bit): [click here](hhttps://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)<br />
    Place the file in the **rl_collaboration_and_competition/** folder, and unzip (or decompress) the file.

# Training

From the home path **rl_collaboration_and_competition/**, run the following command to train the agent:

```bash
python train.py
```

# Visualization

From the home path **rl_collaboration_and_competition/**, run the following command to visualize the trained agent:

```bash
python visualize.py
```

# Video Demo: <br />
[![Video Demo](https://img.youtube.com/vi/OK03tQPBo4g/0.jpg)](https://youtu.be/OK03tQPBo4g)
