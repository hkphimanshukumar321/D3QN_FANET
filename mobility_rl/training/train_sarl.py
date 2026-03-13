import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.sarl_mac_env import AdaptiveMacEnv
from algorithms.rl.sb3_baselines import create_sb3_baseline
from algorithms.rl.tabular_qlearning import TabularQLearning
from algorithms.rl.custom_mca_d3qn import create_mca_d3qn
from configs.sarl_config import RLConfig

def train_sb3_baseline(algo_name):
    print(f"Initializing {algo_name.upper()} Baseline Training...")
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    env = Monitor(AdaptiveMacEnv())
    vec_env = DummyVecEnv([lambda: env])
    
    model = create_sb3_baseline(vec_env, algo_name=algo_name, seed=RLConfig.SEED)
    print(f"Training {algo_name.upper()} for {RLConfig.TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=RLConfig.TOTAL_TIMESTEPS, progress_bar=True)
    
    save_path = os.path.join(RLConfig.get_results_dir(), "checkpoints", f"{algo_name}_baseline_model")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

def train_tabular_baseline():
    print("Initializing Tabular Q-Learning Training...")
    env = AdaptiveMacEnv()
    model = TabularQLearning(seed=RLConfig.SEED)
    obs, info = env.reset(seed=RLConfig.SEED)
    
    from tqdm import tqdm
    pbar = tqdm(total=RLConfig.TOTAL_TIMESTEPS, desc="Training TABULAR", unit="step")
    for step in range(RLConfig.TOTAL_TIMESTEPS):
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        model.learn(obs, action, reward, next_obs, done=(terminated or truncated))
        obs = next_obs
        pbar.update(1)
        if terminated or truncated:
            obs, info = env.reset()
        if (step+1) % 1000 == 0:
            pbar.set_postfix({"Epsilon": f"{model.epsilon:.4f}"})
    pbar.close()
    
    save_path = os.path.join(RLConfig.get_results_dir(), "checkpoints", "tabular_q_model.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

def train_mca_d3qn():
    print("Initializing Custom MCA-D3QN Training...")
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    env = Monitor(AdaptiveMacEnv())
    vec_env = DummyVecEnv([lambda: env])
    
    model = create_mca_d3qn(vec_env, seed=RLConfig.SEED)
    print(f"Training for {RLConfig.TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=RLConfig.TOTAL_TIMESTEPS, progress_bar=True)
    
    save_path = os.path.join(RLConfig.get_results_dir(), "checkpoints", "mca_d3qn_model")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone SARL Trainer")
    parser.add_argument('--algo', type=str, default='all', choices=['all', 'dqn', 'ppo', 'a2c', 'tabular', 'mca_d3qn'])
    args = parser.parse_args()
    
    if args.algo in ['all', 'dqn', 'ppo', 'a2c']:
        baselines = ["dqn", "ppo", "a2c"] if args.algo == 'all' else [args.algo]
        for b in baselines:
            train_sb3_baseline(b)
            
    if args.algo in ['all', 'tabular']:
        train_tabular_baseline()
        
    if args.algo in ['all', 'mca_d3qn']:
        train_mca_d3qn()
