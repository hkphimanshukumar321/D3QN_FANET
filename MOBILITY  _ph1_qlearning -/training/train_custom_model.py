import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.adaptive_mac_env import AdaptiveMacEnv
from algorithms.rl.custom_mca_d3qn import create_mca_d3qn
from configs.rl_config import RLConfig

def main():
    print("Initializing Custom MCA-D3QN Training...")
    env = AdaptiveMacEnv()
    
    # We can wrap in Monitor and DummyVecEnv for standard SB3 eval
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    model = create_mca_d3qn(vec_env, seed=RLConfig.SEED)
    
    # Train the model
    print(f"Training for {RLConfig.TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=RLConfig.TOTAL_TIMESTEPS, progress_bar=True)
    
    # Save the model
    save_path = os.path.join(RLConfig.get_results_dir(), "checkpoints", "mca_d3qn_model")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
