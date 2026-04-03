"""
pipeline/config_loader.py
--------------------------
Loads config.yaml and .env secrets
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Loads project config and injects environment variables.

    Reads .env automatically if present.

    Returns:
        dict: Full config with secrets injected under config["secrets"].
    """
    # Load .env if present
    load_dotenv()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at '{config_path}'. "
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Required secrets
    required = {
        "HF_TOKEN": "HuggingFace token",
        "GROQ_API_KEY": "Groq API key",
        "LANGSMITH_API_KEY": "LangSmith key"
    }
    secrets = {}
    missing = []
    for var, hint in required.items():
        val = os.environ.get(var, "")
        if not val:
            missing.append(f"{var}, {hint}")
        secrets[var] = val

    if missing:
        raise EnvironmentError(
            "Missing required environment variables:\n" + "\n".join(missing)
        )

    '''
    # ChromaDB remote server option
    secrets["CHROMA_HOST"] = os.environ.get("CHROMA_HOST", "")
    secrets["CHROMA_PORT"] = os.environ.get("CHROMA_PORT", "8001")
    '''
    config["secrets"] = secrets

    # Set LangSmith env vars
    os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_API_KEY"] = secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "legal-rag")

    # Create output directories
    for d in ["data", "logs", "evaluation/results", "data/chroma_db"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return config