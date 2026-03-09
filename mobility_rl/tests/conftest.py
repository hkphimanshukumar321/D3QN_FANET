import pytest
import os
import sys

# Ensure the root project directory is in the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def sample_config_path():
    return os.path.join(PROJECT_ROOT, "config.py")

@pytest.fixture
def mock_results_dir(tmp_path):
    """
    Provides a temporary directory for tests to write results into,
    preventing cluttering of the actual `results/` folder during testing.
    """
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    return str(res_dir)
