import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

ray.init()

config = dqn.DEFAULT_CONFIG.copy()

# === 저번주의 테스트와 동일하게 파라미터 설정 ===
config["num_gpus"] = 0 # gpu는 사용하지 않는다.
config["num_workers"] = 1 # 하나의 에이전트만 실행한다.
config["lr"] = 0.001
config["train_batch_size"] = 32
config["exploration_config"] = {
    'type': 'EpsilonGreedy',
    'initial_epsilon': 0.99,
    'final_epsilon': 0.0001,
    'epsilon_timesteps': 10000
}
config["evaluation_duration"] = 4
config["evaluation_duration_unit"] = "timesteps"
config["learning_starts"] = 50000
config["dueling"] = False
config["double_q"] = False
config["buffer_size"] = 100000
config["hiddens"] = [64]
config["framework"] = "torch" # 작성자의 모듈과 동일하게 PyTorch를 사용한다.
#=================================================


trainer = dqn.DQNTrainer(config=config, env="CartPole-v0")

# 한 번의 반복에 1000번의 timestep이 진행된다. 따라서 100 * 1000 = 10만 번의 timestep이 진행된다.
for i in range(100):
    result = trainer.train()
    print(pretty_print(result))
    print("====================================================")
