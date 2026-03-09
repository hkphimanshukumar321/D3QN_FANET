import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.adaptive_mac_env import AdaptiveMacEnv
from algorithms.rl.sb3_baselines import create_sb3_baseline
from algorithms.rl.tabular_qlearning import TabularQLearning
from configs.rl_config import RLConfig

def train_sb3_baseline(algo_name):
    print(f"Initializing {algo_name.upper()} Baseline Training...")
    env = AdaptiveMacEnv()
    
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    env = Monitor(env)
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
    
    # Train using the gym step loop
    for step in range(RLConfig.TOTAL_TIMESTEPS):
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Q-learning update
        model.learn(obs, action, reward, next_obs, done=(terminated or truncated))
        
        obs = next_obs
        
        if terminated or truncated:
            obs, info = env.reset()
            
        if (step+1) % 5000 == 0:
            print(f"Tabular Q-Learning Step: {step+1}/{RLConfig.TOTAL_TIMESTEPS} - Epsilon: {model.epsilon:.4f}")

    save_path = os.path.join(RLConfig.get_results_dir(), "checkpoints", "tabular_q_model.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    baselines = ["dqn", "ppo", "a2c"]
    for b in baselines:
        train_sb3_baseline(b)
        
    train_tabular_baseline()
