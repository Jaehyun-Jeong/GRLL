# 시간 측정
from datetime import datetime
startTime = datetime.now()

import gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v0")

model = DQN("MlpPolicy", env, verbose=0)

print(f"Init Time: {datetime.now() - startTime}")
startTrainTime = datetime.now()

model.learn(total_timesteps=10_0000, n_eval_episodes=0)

print(f"Train Time: {datetime.now() - startTrainTime}")

obs = env.reset()
rewards = 0
returns = []
done_num = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    if done:
      obs = env.reset()
      returns.append(rewards)
      rewards = 0
      done_num += 1
      if done_num == 10:
          break
      
print(returns)

env.close()
