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
    encoding="utf-8",
)

_CHAT_LOGGER = None


def _resolve_chat_logger() -> logging.Logger:
    global _CHAT_LOGGER
    if _CHAT_LOGGER is not None:
        return _CHAT_LOGGER

    chat_cfg = logging_cfg.get("chat_history", {})
    logger = logging.getLogger("chat_history")
    if not logger.handlers:
        filename = chat_cfg.get("filename", "chat_history.log")
        handler = logging.FileHandler(filename, encoding="utf-8")
        formatter = logging.Formatter(
            chat_cfg.get(
                "format",
                logging_cfg.get("format", "%(asctime)s - %(message)s"),
            )
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(chat_cfg.get("level", logging_cfg.get("level", "INFO")))
    logger.propagate = False
    _CHAT_LOGGER = logger
    return logger


def get_app_config():
    """Return the full loaded configuration with environment substitutions applied."""
    return cfg


def get_agent_memory_config():
    """Return only the agent-memory specific section of the configuration."""
    return cfg.get("AgentMemory", {})


def get_hypothesis_config():
    """Return the hypothesis workflow configuration section."""
    return cfg.get("Hypothesis", {})


def get_model_config():
    """Return the model configuration section."""
    return cfg.get("Model", {})


def get_graph_config():
    """Return the graph configuration section."""
    return cfg.get("Graph", {})


def get_chat_logger() -> logging.Logger:
    """Return a dedicated logger for chat history transcripts."""
    return _resolve_chat_logger()


def get_config_section(section_name: str, default=None):
    """Generic helper to retrieve arbitrary config sections."""
    return cfg.get(section_name, {} if default is None else default)