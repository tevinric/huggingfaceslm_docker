import os
from typing import Optional

class Settings:
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # Model Configuration
    MODEL_CONFIG_PATH: str = os.getenv("MODEL_CONFIG_PATH", "app/config/models.yaml")
    DEFAULT_MODEL: Optional[str] = os.getenv("DEFAULT_MODEL", None)
    
    # Cache Settings
    HF_CACHE_DIR: str = os.getenv("HF_HOME", "/app/.cache/huggingface")
    
    # Hugging Face Authentication
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    HF_USERNAME: Optional[str] = os.getenv("HF_USERNAME", None)
    
    # Performance Settings
    MAX_CONCURRENT_MODELS: int = int(os.getenv("MAX_CONCURRENT_MODELS", "3"))
    AUTO_UNLOAD_MINUTES: int = int(os.getenv("AUTO_UNLOAD_MINUTES", "30"))
    
    # Security
    API_KEY: Optional[str] = os.getenv("API_KEY", None)
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()