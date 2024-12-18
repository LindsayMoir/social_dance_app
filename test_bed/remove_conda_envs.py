import os
import subprocess

def get_conda_envs():
    """Fetch the list of all conda environments."""
    try:
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
        envs = []
        for line in result.stdout.splitlines():
            # Environments are listed with their full paths; filter them
            if line.strip() and "/" in line:
                envs.append(line.split()[0])  # Extracts the environment path
        print('List of environments found')
        print(envs)
        return envs
    except subprocess.CalledProcessError as e:
        print(f"Error fetching conda environments: {e}")
        return []

def delete_conda_env(env_path):
    """Delete a conda environment by its name or path."""
    try:
        subprocess.run(["conda", "env", "remove", "--name", env_path], check=True)
        print(f"Deleted environment: {env_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to delete environment {env_path}: {e}")

def delete_mlflow_envs():
    """Delete all conda environments containing the word 'mlflow'."""
    envs = get_conda_envs()
    mlflow_envs = [env for env in envs if 'mlflow' in os.path.basename(env)]

    if not mlflow_envs:
        print("No environments found containing 'mlflow'.")
        return

    for env in mlflow_envs:
        delete_conda_env(env)

if __name__ == "__main__":
    delete_mlflow_envs()
