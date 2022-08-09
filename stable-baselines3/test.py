import gym

from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1,
            buffer_size=10000,
            batch_size=100,
            train_freq=(1, 'step'),
            learning_starts=1)

model.learn(total_timesteps=10_0000, n_eval_episodes=0, eval_freq=1)

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
