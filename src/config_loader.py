import yaml
import os

def load_config(config_filename: str = "model_config.yaml") -> dict:
    """
    Loads YAML config from the config/ directory.
    Resolves path dynamically — works in Databricks Repos.
    
    Args:
        config_filename: Name of the YAML file to load
    Returns:
        dict: Parsed configuration dictionary
    """
    # src/ is here, config/ is one level up then into config/
    src_dir     = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    config_path  = os.path.join(project_root, "config", config_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            f"Project root resolved to: {project_root}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f" Config loaded: {config_path}")
    return config


# Module-level singleton — loaded once, reused everywhere
MODEL_CONFIG    = load_config("model_config.yaml")
PIPELINE_CONFIG = load_config("pipeline_config.yaml")
FEATURE_CONFIG  = load_config("feature_config.yaml")
