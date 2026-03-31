import pytest
import subprocess
import os
import sys

@pytest.mark.static
def test_generate_dead_code_report(project_root, mock_results_dir):
    """
    Static Code Analysis:
    Runs `vulture` programmatically across the repository to identify
    functions, variables, or imports that are never executed.
    
    This test doesn't fail on finding dead code (since research repos
    often have legacy scripts), but strictly guarantees the report
    is successfully generated for manual review.
    """
    target_dirs = [
        os.path.join(project_root, "algorithms"),
        os.path.join(project_root, "simulator_ui"),
        os.path.join(project_root, "config.py")
    ]
    
    report_path = os.path.join(mock_results_dir, "dead_code_report.txt")
    
    # We pass the target directories to the Vulture CLI tool using python -m
    # to completely bypass any Linux $PATH variable issues with virtualenvs
    cmd = [sys.executable, "-m", "vulture"] + target_dirs
    
    try:
        # Vulture exits with 1 if it finds dead code, which is expected.
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        pytest.fail(f"Execution of vulture failed: {e}")
        
    with open(report_path, "w") as f:
        f.write("--- INTERNAL VULTURE STATIC DEAD CODE REPORT ---\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- ERRORS ---\n")
            f.write(result.stderr)
            
    assert os.path.exists(report_path), "Failed to generate the dead code report."
    
    # Verify the report has content (even if the content says 0 issues, vulture prints output)
    with open(report_path, "r") as f:
        content = f.read()
        assert len(content) > 10, "Dead code report is suspiciously empty."
