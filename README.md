# GRLL

| **Algorithm** | **Discrete Action Space** | **Continuous Action Space** | 
| ------------- | ------------------------- | --------------------------- |
| A2C | :heavy_check_mark: | :heavy_check_mark: |
| REINFORCE | :heavy_check_mark: | :heavy_check_mark: |
| DQN | :heavy_check_mark: | :x: |
| ADQN | :heavy_check_mark: | :x: |

## Install

## Usage

Below is an example tested using OpenAI's gym.
```python
import torch.optim as optim

# Import the implemented module and the default neural network model
from grll.PG.models import ANN_V2
from grll.PG import A2C

# Environment
import gymnasium as gym
env = gym.make('CartPole-v0')

# Create the neural network
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
A2C_model = ANN_V2(num_states, num_actions)

# Create the optimizer
optimizer = optim.Adam(A2C_model.parameters(), lr=1e-4)

# Initialize the reinforcement learning class
advantage_AC = A2C(
    env=env,
    model=A2C_model,
    optimizer=optimizer,
)

# Train the model
advantage_AC.train(trainTimesteps=1000000)

# Save the class
ADeepQLearning.save("./saved_models/test.obj")

```

If you want to use a different algorithm, you can write it like this:<br/>
```python
from GRLL.PG import REINFORCE
"""
Or
from GRLL.VB import DQN
from GRLL.VB import ADQN
"""
```

## Custom Environment

If you have downloaded the pygame module, you can use the following two environments.

### RacingEnv

![](/static/RacingEnv.png)

[NeuralNine](https://www.youtube.com/watch?v=Cy155O5R1Oo&t=527s&ab_channel=NeuralNine)

RacingEnv_v0: Receives the lengths of 5 sensors as the state and has four actions: right, left, acceleration, and brake.<br/>

### MazeEnv

![](/static/MazeEnv.png)

MazeEnv_v0: Receives vector information of the entire map as the state and has four actions to move north, south, east, and west.<br/>
