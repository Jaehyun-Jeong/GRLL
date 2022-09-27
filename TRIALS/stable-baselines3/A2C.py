from datetime import datetime

# 시작 시간
startTime = datetime.now()

import gym
from stable_baselines3 import A2C

env = gym.make("CartPole-v0")

# 작성자의 모듈과 동일하게 파라미터 설정
model = A2C(
        "MlpPolicy",
        env,
        gae_lambda=1,
        use_rms_prop=False,  # RMSProp 옵티마이저 대신에 Adam을 사용
        learning_rate=1e-4,
        n_steps=10,
        vf_coef=1,
        verbose=0)

# 모듈 초기화에 걸린 시간
print(f"Init Time: {datetime.now() - startTime}")

# 학습이 시작되는 시간
startTrainTime = datetime.now()

model.learn(
        total_timesteps=1000000,  # 백만번의 학습을 시행
        n_eval_episodes=0)

# 학습이 끝나는 시간
print(f"Train Time: {datetime.now() - startTrainTime}")

# 성능 측정을 위한 테스트 코드
returns = [0]*10
for episode in range(10):
    obs = env.reset()
    while True:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        returns[episode] += reward
        if done:
            break

print(sum(returns) / 10)
print("=================================================")
