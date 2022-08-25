from datetime import datetime
startTime = datetime.now()

from stable_baselines3 import A2C

# Parallel environments
env = "CartPole-v0"

model = A2C(
        "MlpPolicy",
        env,
        gae_lambda=1,
        use_rms_prop=False,  # RMSProp 옵티마이저 대신에 Adam을 사용
        learning_rate=1e-4,
        n_steps=10,
        vf_coef=1,
        verbose=0)

print(f"Init Time: {datetime.now() - startTime}")
startTrainTime = datetime.now()

model.learn(
        total_timesteps=1000000,  # 백만번의 학습을 시행
        n_eval_episodes=0)

print(f"Train Time: {datetime.now() - startTrainTime}")

# Test
returns = [0]*10
for episode in range(10):
    obs = env.reset()
    while True:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        returns[episode] += reward
        if done:
            break

print(returns)
print("=================================================")
