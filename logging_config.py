# initialize logging configuration
import logging
import yaml
import dotenv
import os

dotenv.load_dotenv()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Substitute environmental variables in cfg if needed
def substitute_env_vars(obj):
    if isinstance(obj, dict):
        return {k: substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars(i) for i in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_var = obj[2:-1]
        return os.getenv(env_var, obj)
    else:
        return obj

cfg = substitute_env_vars(cfg)


logging_cfg = cfg.get("Logging", {})
logging.basicConfig(
    filename=logging_cfg.get("filename", "workflow.log"),
    level=logging_cfg.get("level", "WARNING"),
    format=logging_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
)
def get_app_config():
    """Return the full loaded configuration with environment substitutions applied."""
    return cfg


def get_agent_memory_config():
    """Return only the agent-memory specific section of the configuration."""
    return cfg.get("AgentMemory", {})


def get_hypothesis_config():
    """Return the hypothesis workflow configuration section."""
    return cfg.get("Hypothesis", {})


def get_config_section(section_name: str, default=None):
    """Generic helper to retrieve arbitrary config sections."""
    return cfg.get(section_name, {} if default is None else default)