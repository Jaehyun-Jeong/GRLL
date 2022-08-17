# 시간 측정
from datetime import datetime

# 시작 시간
startTime = datetime.now()

import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

ray.init()

# 모듈 초기화에 걸린 시간
print(f"Init Time: {datetime.now() - startTime}")

# === 저번주의 테스트와 동일하게 파라미터 설정 ===
config = dqn.DEFAULT_CONFIG.copy()
# gpu는 사용하지 않는다.
config["num_gpus"] = 0
# 하나의 에이전트만 실행한다.
config["num_workers"] = 1
config["lr"] = 0.001
config["train_batch_size"] = 32
config["exploration_config"] = {
    'type': 'EpsilonGreedy',
    'initial_epsilon': 0.99,
    'final_epsilon': 0.0001,
    'epsilon_timesteps': 10000
}
config["dueling"] = False
config["double_q"] = False
config["replay_buffer_config"]["capacity"] = 100000
config["replay_buffer_config"]["learning_starts"] = 50000
config["model"]["fcnet_hiddens"] = [64]
config["model"]["fcnet_activation"] = 'relu'
# 작성자의 모듈과 동일하게 PyTorch를 사용한다.
config["framework"] = "torch"
# ================================================

# 학습이 시작되는 시간
startTrainTime = datetime.now()

trainer = dqn.DQNTrainer(config=config, env="CartPole-v0")

# 한 번의 반복에 1000번의 timestep이 진행된다. 따라서 100 * 1000 = 10만 번의 timestep이 진행된다.
for i in range(100):
    if i < 99:
        trainer.train()
    else:
        result = trainer.train()

# 학습이 끝나는 시간
print(f"Train Time: {datetime.now() - startTrainTime}")

# 마지막 학습에서 얻어진 결과 출력
print(pretty_print(result))
