import pytest
import sys
import importlib

@pytest.mark.smoke
def test_repository_imports():
    """
    Verifies that all major packages and modules can be imported without
    SyntaxErrors or ModuleNotFound errors.
    """
    modules_to_check = [
        "configs.config",
        "configs.sarl_config",
        "configs.marl_config",
        "algorithms.mac.baseline",
        "algorithms.mobility.models",
        "algorithms.mobility.speed",
        "algorithms.mobility.link",
        "algorithms.rl.qlearning_selector",
        "algorithms.rl.marl_baselines",
        "algorithms.rl.gnn_marl",
        "algorithms.channel.fading",
        "envs.marl_mac_env",
        "simulator_ui.engine",
        "simulator_ui.server",
    ]
    
    for mod_name in modules_to_check:
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            pytest.fail(f"Failed to import core module '{mod_name}': {e}")

