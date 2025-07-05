# Complete Multi-Model Hugging Face API Guide

## Overview

This guide provides a production-ready implementation for serving multiple Hugging Face Small Language Models (SLMs) through Docker containers with dynamic model loading, configuration-based setup, and automatic endpoint generation.

## Key Features

- **🔄 Dynamic Model Loading**: Load/unload models on-demand via API
- **📝 Configuration-Based**: Add new models by editing YAML (no code changes)
- **🚀 Auto Endpoints**: Every model gets generate/chat/load/unload endpoints automatically
- **💾 Smart Memory Management**: Automatic cleanup and memory optimization
- **🐳 Docker Ready**: Models cached outside container, small image size
- **🔧 Production Features**: Health checks, logging, error handling, monitoring

## Project Structure

Create this complete directory structure:

```
huggingface-multi-model-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_manager.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── models.yaml
│   │   └── settings.py
│   └── utils.py
├── client_examples/
│   ├── basic_client.py
│   ├── multi_model_client.py
│   └── async_client.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Complete Implementation

### 1. Requirements File

**`requirements.txt`**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.35.2
torch==2.1.1
tokenizers==0.15.0
accelerate==0.25.0
pydantic==2.5.0
python-multipart==0.0.6
PyYAML==6.0.1
numpy==1.24.3
safetensors==0.4.1
huggingface-hub==0.19.4
```

### 2. Model Configuration

**`app/config/models.yaml`**
```yaml
# Model Registry Configuration
models:
  # === Small Conversational Models ===
  dialogpt-small:
    name: "microsoft/DialoGPT-small"
    type: "conversational"
    size: "117M"
    description: "Small conversational model for chat applications"
    max_length: 1000
    suggested_temperature: 0.7

  dialogpt-medium:
    name: "microsoft/DialoGPT-medium"
    type: "conversational"
    size: "345M"
    description: "Medium conversational model with better responses"
    max_length: 1000
    suggested_temperature: 0.7

  # === Text Generation Models ===
  gpt2-small:
    name: "gpt2"
    type: "text"
    size: "124M"
    description: "GPT-2 small model for general text generation"
    max_length: 1024
    suggested_temperature: 0.8

  distilgpt2:
    name: "distilgpt2"
    type: "text"
    size: "82M"
    description: "Distilled GPT-2 model - faster and smaller"
    max_length: 1024
    suggested_temperature: 0.8

  gpt2-medium:
    name: "gpt2-medium"
    type: "text"
    size: "345M"
    description: "GPT-2 medium model for better text quality"
    max_length: 1024
    suggested_temperature: 0.8

  # === Code Generation Models ===
  codegen-350m:
    name: "Salesforce/codegen-350M-mono"
    type: "code"
    size: "350M"
    description: "Code generation model for Python/programming"
    max_length: 2048
    suggested_temperature: 0.2

  codegen-2b:
    name: "Salesforce/codegen-2B-mono"
    type: "code"
    size: "2B"
    description: "Larger code generation model"
    max_length: 2048
    suggested_temperature: 0.2

  # === Instruction Following Models ===
  flan-t5-small:
    name: "google/flan-t5-small"
    type: "instruction"
    size: "80M"
    description: "Small instruction-following model"
    max_length: 512
    suggested_temperature: 0.3

  flan-t5-base:
    name: "google/flan-t5-base"
    type: "instruction"
    size: "250M"
    description: "Base instruction-following model"
    max_length: 512
    suggested_temperature: 0.3

  # === Lightweight Models ===
  opt-125m:
    name: "facebook/opt-125m"
    type: "text"
    size: "125M"
    description: "OPT 125M parameter model"
    max_length: 2048
    suggested_temperature: 0.8

  opt-350m:
    name: "facebook/opt-350m"
    type: "text"
    size: "350M"
    description: "OPT 350M parameter model"
    max_length: 2048
    suggested_temperature: 0.8

  # === Specialized Models ===
  pythia-160m:
    name: "EleutherAI/pythia-160m"
    type: "text"
    size: "160M"
    description: "Pythia model for text generation"
    max_length: 2048
    suggested_temperature: 0.8

  bloom-560m:
    name: "bigscience/bloom-560m"
    type: "text"
    size: "560M"
    description: "BLOOM multilingual model"
    max_length: 1024
    suggested_temperature: 0.8

  # === GATED MODELS (Require HF_TOKEN) ===
  llama-3.2-1b:
    name: "meta-llama/Llama-3.2-1B"
    type: "text"
    size: "1B"
    description: "Llama 3.2 1B model (GATED - requires HF token)"
    max_length: 2048
    suggested_temperature: 0.7
    gated: true

  llama-3.2-3b:
    name: "meta-llama/Llama-3.2-3B"
    type: "text"
    size: "3B"
    description: "Llama 3.2 3B model (GATED - requires HF token)"
    max_length: 2048
    suggested_temperature: 0.7
    gated: true

  llama-3.2-1b-instruct:
    name: "meta-llama/Llama-3.2-1B-Instruct"
    type: "instruction"
    size: "1B"
    description: "Llama 3.2 1B Instruct model (GATED - requires HF token)"
    max_length: 2048
    suggested_temperature: 0.6
    gated: true

  # === Other Gated Models ===
  gemma-2b:
    name: "google/gemma-2b"
    type: "text"
    size: "2B"
    description: "Gemma 2B model (GATED - requires HF token)"
    max_length: 8192
    suggested_temperature: 0.7
    gated: true

# System Settings
settings:
  # Memory Management
  max_concurrent_models: 3
  auto_unload_after_minutes: 30
  memory_cleanup_interval_minutes: 5
  
  # Model Loading
  default_device: "auto"
  cache_dir: "/app/.cache/huggingface"
  low_cpu_mem_usage: true
  torch_dtype: "auto"
  
  # Performance
  use_fast_tokenizer: true
  trust_remote_code: false
  
  # API Settings
  default_max_length: 100
  max_batch_size: 4
  
  # Logging
  log_level: "INFO"
  log_model_loading: true
```

### 3. Settings Configuration

**`app/config/settings.py`**
```python
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
```

**`app/model_manager.py`**
```python
import torch
import yaml
import asyncio
import logging
import threading
from typing import Dict, Optional, List, Any, Union
from datetime import datetime, timedelta
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    pipeline
)
from pathlib import Path
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ModelInfo:
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.name = config["name"]
        self.type = config["type"]
        self.size = config["size"]
        self.description = config["description"]
        self.max_length = config["max_length"]
        self.suggested_temperature = config.get("suggested_temperature", 0.7)
        self.gated = config.get("gated", False)
        
        # Runtime state
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.last_used = None
        self.loading = False
        self.load_lock = threading.Lock()
        self.device = None

class ModelManager:
    def __init__(self, config_path: str = "app/config/models.yaml"):
        self.models: Dict[str, ModelInfo] = {}
        self.settings = {}
        self.config_path = config_path
        self._cleanup_task = None
        self._load_config()
        
    def _load_config(self):
        """Load model configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Config file not found: {self.config_path}")
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                
            self.settings = config.get("settings", {})
            
            # Load model configurations
            models_config = config.get("models", {})
            for model_id, model_config in models_config.items():
                self.models[model_id] = ModelInfo(model_id, model_config)
                
            logger.info(f"Loaded {len(self.models)} model configurations")
            
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            raise
    
    def reload_config(self):
        """Reload configuration without restarting the service"""
        logger.info("Reloading model configuration...")
        old_models = set(self.models.keys())
        self._load_config()
        new_models = set(self.models.keys())
        
        added = new_models - old_models
        removed = old_models - new_models
        
        if added:
            logger.info(f"Added models: {', '.join(added)}")
        if removed:
            logger.info(f"Removed models: {', '.join(removed)}")
            # Unload removed models
            for model_id in removed:
                asyncio.create_task(self.unload_model(model_id))
    
    def _get_auth_token(self) -> Optional[str]:
        """Get Hugging Face authentication token"""
        return settings.HF_TOKEN
    
    def _prepare_model_kwargs(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Prepare keyword arguments for model loading"""
        kwargs = {
            "cache_dir": self.settings.get("cache_dir"),
            "low_cpu_mem_usage": self.settings.get("low_cpu_mem_usage", True),
            "trust_remote_code": self.settings.get("trust_remote_code", False)
        }
        
        # Add authentication token if available
        token = self._get_auth_token()
        if token:
            kwargs["token"] = token
            logger.info(f"Using HF token for model {model_info.model_id}")
        elif model_info.gated:
            logger.warning(f"Model {model_info.model_id} is gated but no HF token provided")
        
        # Add device and dtype configuration
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
            
        return kwargs
    
    async def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """Load a specific model"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in registry")
            
        model_info = self.models[model_id]
        
        # Check if already loaded and not forcing reload
        if not force_reload and model_info.model is not None:
            model_info.last_used = datetime.now()
            logger.info(f"Model {model_id} already loaded")
            return True
        
        # Check for gated model without token
        if model_info.gated and not self._get_auth_token():
            error_msg = (
                f"Model '{model_id}' is gated and requires a Hugging Face token. "
                f"Please set HF_TOKEN environment variable or request access at "
                f"https://huggingface.co/{model_info.name}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Handle concurrent loading
        with model_info.load_lock:
            if model_info.loading:
                # Wait for loading to complete
                while model_info.loading:
                    await asyncio.sleep(0.1)
                return model_info.model is not None
                
            model_info.loading = True
            
        try:
            logger.info(f"Loading model: {model_info.name} ({model_info.size})")
            if model_info.gated:
                logger.info(f"Loading gated model with authentication")
            
            # Manage memory before loading
            await self._manage_memory()
            
            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
                model_info.device = "cuda"
            else:
                device = "cpu"
                model_info.device = "cpu"
            
            # Prepare loading arguments
            model_kwargs = self._prepare_model_kwargs(model_info)
            tokenizer_kwargs = {
                "cache_dir": self.settings.get("cache_dir"),
                "use_fast": self.settings.get("use_fast_tokenizer", True),
                "trust_remote_code": self.settings.get("trust_remote_code", False)
            }
            
            # Add token for tokenizer if available
            token = self._get_auth_token()
            if token:
                tokenizer_kwargs["token"] = token
            
            # Load tokenizer
            model_info.tokenizer = AutoTokenizer.from_pretrained(
                model_info.name,
                **tokenizer_kwargs
            )
            
            # Add padding token if needed
            if model_info.tokenizer.pad_token is None:
                model_info.tokenizer.pad_token = model_info.tokenizer.eos_token
            
            # Choose appropriate model class
            if model_info.type == "instruction":
                ModelClass = AutoModelForSeq2SeqLM
            else:
                ModelClass = AutoModelForCausalLM
            
            # Load model
            model_info.model = ModelClass.from_pretrained(
                model_info.name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not torch.cuda.is_available():
                model_info.model = model_info.model.to(device)
            
            model_info.last_used = datetime.now()
            model_info.loading = False
            
            logger.info(f"Model {model_id} loaded successfully on {model_info.device}")
            return True
            
        except Exception as e:
            model_info.loading = False
            logger.error(f"Error loading model {model_id}: {e}")
            
            # Provide helpful error message for gated models
            if "gated" in str(e).lower() or "token" in str(e).lower():
                helpful_error = (
                    f"Failed to load gated model '{model_id}'. "
                    f"Please ensure you have:\n"
                    f"1. Requested access at https://huggingface.co/{model_info.name}\n"
                    f"2. Set HF_TOKEN environment variable with your token\n"
                    f"3. Your token has permission to access this model\n"
                    f"Original error: {str(e)}"
                )
                raise ValueError(helpful_error)
            
            raise
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a specific model to free memory"""
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found for unloading")
            return False
            
        model_info = self.models[model_id]
        
        if model_info.model is not None:
            # Clean up model resources
            del model_info.model
            del model_info.tokenizer
            if model_info.pipeline:
                del model_info.pipeline
                
            model_info.model = None
            model_info.tokenizer = None
            model_info.pipeline = None
            model_info.device = None
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"Model {model_id} unloaded successfully")
            return True
        
        return False
    
    async def _manage_memory(self):
        """Manage memory by unloading old models if needed"""
        loaded_models = [
            (mid, info) for mid, info in self.models.items() 
            if info.model is not None
        ]
        
        max_models = self.settings.get("max_concurrent_models", 3)
        
        if len(loaded_models) >= max_models:
            # Sort by last used time and unload oldest
            loaded_models.sort(key=lambda x: x[1].last_used or datetime.min)
            
            models_to_unload = len(loaded_models) - max_models + 1
            for i in range(models_to_unload):
                model_id = loaded_models[i][0]
                await self.unload_model(model_id)
                logger.info(f"Auto-unloaded model {model_id} to free memory")
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info and update last used time"""
        if model_id in self.models and self.models[model_id].model is not None:
            self.models[model_id].last_used = datetime.now()
            return self.models[model_id]
        return None
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their status"""
        return {
            model_id: {
                "name": info.name,
                "type": info.type,
                "size": info.size,
                "description": info.description,
                "max_length": info.max_length,
                "suggested_temperature": info.suggested_temperature,
                "gated": info.gated,
                "loaded": info.model is not None,
                "loading": info.loading,
                "device": info.device,
                "last_used": info.last_used.isoformat() if info.last_used else None
            }
            for model_id, info in self.models.items()
        }
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs"""
        return [
            model_id for model_id, info in self.models.items() 
            if info.model is not None
        ]
    
    def get_models_by_type(self, model_type: str) -> List[str]:
        """Get models filtered by type"""
        return [
            model_id for model_id, info in self.models.items()
            if info.type == model_type
        ]
    
    def get_gated_models(self) -> List[str]:
        """Get list of gated models"""
        return [
            model_id for model_id, info in self.models.items()
            if info.gated
        ]
    
    def check_auth_status(self) -> Dict[str, Any]:
        """Check authentication status for gated models"""
        token = self._get_auth_token()
        gated_models = self.get_gated_models()
        
        return {
            "has_token": token is not None,
            "token_preview": f"{token[:8]}..." if token else None,
            "gated_models_count": len(gated_models),
            "gated_models": gated_models
        }
    
    async def start_cleanup_task(self):
        """Start background task to cleanup unused models"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_loop(self):
        """Background task to unload unused models"""
        auto_unload_minutes = self.settings.get("auto_unload_after_minutes", 30)
        cleanup_interval = self.settings.get("memory_cleanup_interval_minutes", 5)
        
        while True:
            try:
                await asyncio.sleep(cleanup_interval * 60)  # Convert to seconds
                
                cutoff_time = datetime.now() - timedelta(minutes=auto_unload_minutes)
                models_to_unload = [
                    model_id for model_id, info in self.models.items()
                    if (info.model is not None and 
                        info.last_used and 
                        info.last_used < cutoff_time)
                ]
                
                for model_id in models_to_unload:
                    await self.unload_model(model_id)
                    logger.info(f"Auto-unloaded inactive model: {model_id}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

# Global model manager instance
model_manager = ModelManager()
```

### 5. Utility Functions

**`app/utils.py`**
```python
import torch
import psutil
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        "python_version": "",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": [],
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "memory_used_percent": psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        
        # Get info for each GPU
        for i in range(torch.cuda.device_count()):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": torch.cuda.get_device_properties(i).total_memory // (1024**2),
                "memory_allocated_mb": torch.cuda.memory_allocated(i) // (1024**2),
                "memory_cached_mb": torch.cuda.memory_reserved(i) // (1024**2)
            }
            info["cuda_devices"].append(device_info)
    
    return info

def get_model_memory_usage(model) -> Dict[str, Any]:
    """Estimate model memory usage"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory (rough calculation)
        param_size_mb = total_params * 4 / (1024**2)  # Assuming float32
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": round(param_size_mb, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def validate_generation_params(
    max_length: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    model_max_length: int = 2048
) -> Dict[str, Any]:
    """Validate and adjust generation parameters"""
    params = {}
    
    # Validate max_length
    if max_length is not None:
        params["max_length"] = min(max_length, model_max_length)
    else:
        params["max_length"] = min(100, model_max_length)
    
    # Validate temperature
    if temperature is not None:
        params["temperature"] = max(0.1, min(temperature, 2.0))
    else:
        params["temperature"] = 0.7
    
    # Validate top_p
    if top_p is not None:
        params["top_p"] = max(0.1, min(top_p, 1.0))
    else:
        params["top_p"] = 0.9
    
    # Validate top_k
    if top_k is not None:
        params["top_k"] = max(1, min(top_k, 100))
    else:
        params["top_k"] = 50
    
    return params
```

### 6. Main Application

**`app/main.py`**
```python
from fastapi import FastAPI, HTTPException, Path, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import logging
import asyncio
from contextlib import asynccontextmanager

from app.model_manager import model_manager
from app.config.settings import settings
from app.utils import get_system_info, get_model_memory_usage, clear_gpu_memory, validate_generation_params

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class TextGenerationRequest(BaseModel):
    text: str = Field(..., description="Input text to generate from")
    max_length: Optional[int] = Field(None, ge=1, le=4096, description="Maximum length of generated text")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-k sampling")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    num_return_sequences: Optional[int] = Field(1, ge=1, le=5, description="Number of sequences to return")

class TextGenerationResponse(BaseModel):
    generated_text: List[str]
    model_id: str
    model_name: str
    input_length: int
    output_length: List[int]
    generation_params: Dict[str, Any]

class ModelLoadRequest(BaseModel):
    force_reload: Optional[bool] = Field(False, description="Force reload even if already loaded")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Chat message")
    max_length: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.1, le=2.0)
    context: Optional[str] = Field(None, description="Previous conversation context")

class ChatResponse(BaseModel):
    response: str
    model_id: str
    model_name: str
    context: Optional[str] = None

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Multi-Model Hugging Face API...")
    await model_manager.start_cleanup_task()
    
    # Load default model if specified
    if settings.DEFAULT_MODEL:
        try:
            await model_manager.load_model(settings.DEFAULT_MODEL)
            logger.info(f"Default model {settings.DEFAULT_MODEL} loaded")
        except Exception as e:
            logger.warning(f"Failed to load default model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await model_manager.stop_cleanup_task()

# Create FastAPI app
app = FastAPI(
    title="Multi-Model Hugging Face API",
    description="Production-ready API for serving multiple Hugging Face models with dynamic loading",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API key dependency
async def verify_api_key(api_key: Optional[str] = None):
    if settings.API_KEY and api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Root endpoints
@app.get("/")
async def root():
    return {
        "message": "Multi-Model Hugging Face API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    loaded_models = model_manager.get_loaded_models()
    system_info = get_system_info()
    
    return {
        "status": "healthy",
        "loaded_models": loaded_models,
        "total_models_available": len(model_manager.models),
        "system_info": {
            "cuda_available": system_info["cuda_available"],
            "memory_used_percent": system_info["memory_used_percent"],
            "cuda_devices": len(system_info.get("cuda_devices", []))
        }
    }

@app.get("/system")
async def get_system_info_endpoint():
    """Get detailed system information"""
    return get_system_info()

# Model management endpoints
@app.get("/models")
async def list_models():
    """List all available models with their current status"""
    return model_manager.list_models()

@app.get("/models/loaded")
async def list_loaded_models():
    """Get currently loaded models"""
    return {"loaded_models": model_manager.get_loaded_models()}

@app.get("/models/types/{model_type}")
async def get_models_by_type(model_type: str = Path(..., description="Model type to filter by")):
    """Get models filtered by type (text, conversational, code, instruction)"""
    models = model_manager.get_models_by_type(model_type)
    if not models:
        raise HTTPException(status_code=404, detail=f"No models found for type: {model_type}")
    return {"model_type": model_type, "models": models}

@app.post("/models/{model_id}/load")
async def load_model(
    model_id: str = Path(..., description="Model ID to load"),
    request: Optional[ModelLoadRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: bool = Depends(verify_api_key)
):
    """Load a specific model"""
    force_reload = request.force_reload if request else False
    
    try:
        success = await model_manager.load_model(model_id, force_reload=force_reload)
        if success:
            return {
                "message": f"Model {model_id} loaded successfully",
                "model_id": model_id,
                "force_reload": force_reload
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/models/{model_id}/unload")
async def unload_model(
    model_id: str = Path(..., description="Model ID to unload"),
    _: bool = Depends(verify_api_key)
):
    """Unload a specific model to free memory"""
    success = await model_manager.unload_model(model_id)
    if success:
        return {"message": f"Model {model_id} unloaded successfully"}
    else:
        return {"message": f"Model {model_id} was not loaded"}

# Authentication endpoints
@app.get("/auth/status")
async def get_auth_status():
    """Get authentication status for gated models"""
    return model_manager.check_auth_status()

@app.get("/models/gated")
async def list_gated_models():
    """List all gated models"""
    gated_models = model_manager.get_gated_models()
    auth_status = model_manager.check_auth_status()
    
    return {
        "gated_models": gated_models,
        "auth_status": auth_status,
        "models_detail": {
            model_id: model_manager.models[model_id].__dict__ 
            for model_id in gated_models
        }
    }

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str = Path(..., description="Model ID")):
    """Get detailed information about a specific model"""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = model_manager.models[model_id]
    response = {
        "model_id": model_id,
        "name": model_info.name,
        "type": model_info.type,
        "size": model_info.size,
        "description": model_info.description,
        "max_length": model_info.max_length,
        "suggested_temperature": model_info.suggested_temperature,
        "gated": model_info.gated,
        "loaded": model_info.model is not None,
        "loading": model_info.loading,
        "device": model_info.device,
        "last_used": model_info.last_used.isoformat() if model_info.last_used else None
    }
    
    # Add auth status for gated models
    if model_info.gated:
        auth_status = model_manager.check_auth_status()
        response["auth_status"] = auth_status["has_token"]
        response["access_url"] = f"https://huggingface.co/{model_info.name}"
    
    # Add memory usage if model is loaded
    if model_info.model is not None:
        response["memory_usage"] = get_model_memory_usage(model_info.model)
    
    return response

# Text generation endpoints
@app.post("/models/{model_id}/generate", response_model=TextGenerationResponse)
async def generate_text_with_model(
    request: TextGenerationRequest,
    model_id: str = Path(..., description="Model ID to use for generation"),
    _: bool = Depends(verify_api_key)
):
    """Generate text using a specific model"""
    
    # Validate model exists
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get or load model
    model_info = model_manager.get_model(model_id)
    if model_info is None:
        try:
            await model_manager.load_model(model_id)
            model_info = model_manager.get_model(model_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        if model_info is None:
            raise HTTPException(status_code=503, detail="Model failed to load")
    
    try:
        # Validate and adjust parameters
        gen_params = validate_generation_params(
            request.max_length, 
            request.temperature, 
            request.top_p, 
            request.top_k,
            model_info.max_length
        )
        
        # Tokenize input
        inputs = model_info.tokenizer.encode(request.text, return_tensors="pt")
        device = next(model_info.model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model_info.model.generate(
                inputs,
                max_length=gen_params["max_length"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                top_k=gen_params["top_k"],
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=model_info.tokenizer.eos_token_id,
                eos_token_id=model_info.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        output_lengths = []
        
        for output in outputs:
            decoded = model_info.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(decoded)
            output_lengths.append(len(output))
        
        return TextGenerationResponse(
            generated_text=generated_texts,
            model_id=model_id,
            model_name=model_info.name,
            input_length=len(inputs[0]),
            output_length=output_lengths,
            generation_params=gen_params
        )
        
    except Exception as e:
        logger.error(f"Error generating text with model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.post("/models/{model_id}/chat", response_model=ChatResponse)
async def chat_with_model(
    request: ChatRequest,
    model_id: str = Path(..., description="Model ID to use for chat"),
    _: bool = Depends(verify_api_key)
):
    """Chat with a specific model"""
    
    # Get model info
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = model_manager.get_model(model_id)
    if model_info is None:
        await model_manager.load_model(model_id)
        model_info = model_manager.get_model(model_id)
        
        if model_info is None:
            raise HTTPException(status_code=503, detail="Failed to load model")
    
    try:
        # Format input based on model type
        if model_info.type == "conversational":
            if request.context:
                formatted_input = f"{request.context}\nHuman: {request.message}\nAssistant:"
            else:
                formatted_input = f"Human: {request.message}\nAssistant:"
        else:
            formatted_input = request.message
        
        # Create generation request
        gen_request = TextGenerationRequest(
            text=formatted_input,
            max_length=request.max_length or model_info.max_length,
            temperature=request.temperature or model_info.suggested_temperature,
            num_return_sequences=1
        )
        
        # Generate response
        response = await generate_text_with_model(gen_request, model_id)
        
        # Extract response based on model type
        generated_text = response.generated_text[0]
        if model_info.type == "conversational" and "Assistant:" in generated_text:
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            # Update context
            new_context = formatted_input + " " + assistant_response
        else:
            assistant_response = generated_text
            new_context = request.context
        
        return ChatResponse(
            response=assistant_response,
            model_id=model_id,
            model_name=model_info.name,
            context=new_context
        )
        
    except Exception as e:
        logger.error(f"Error in chat with model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

# Convenience endpoints
@app.post("/generate")
async def default_generate(request: TextGenerationRequest, _: bool = Depends(verify_api_key)):
    """Generate text using the first available text generation model"""
    text_models = model_manager.get_models_by_type("text")
    if not text_models:
        raise HTTPException(status_code=404, detail="No text generation models available")
    
    return await generate_text_with_model(request, text_models[0])

@app.post("/chat")
async def default_chat(request: ChatRequest, _: bool = Depends(verify_api_key)):
    """Chat using the first available conversational model"""
    conv_models = model_manager.get_models_by_type("conversational")
    if not conv_models:
        raise HTTPException(status_code=404, detail="No conversational models available")
    
    return await chat_with_model(request, conv_models[0])

# Administrative endpoints
@app.post("/admin/reload-config")
async def reload_config(_: bool = Depends(verify_api_key)):
    """Reload model configuration without restarting"""
    try:
        model_manager.reload_config()
        return {"message": "Configuration reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading config: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading config: {str(e)}")

@app.post("/admin/clear-gpu-memory")
async def clear_gpu_memory_endpoint(_: bool = Depends(verify_api_key)):
    """Clear GPU memory cache"""
    clear_gpu_memory()
    return {"message": "GPU memory cache cleared"}

@app.post("/admin/unload-all")
async def unload_all_models(_: bool = Depends(verify_api_key)):
    """Unload all currently loaded models"""
    loaded_models = model_manager.get_loaded_models()
    
    for model_id in loaded_models:
        await model_manager.unload_model(model_id)
    
    return {
        "message": f"Unloaded {len(loaded_models)} models",
        "unloaded_models": loaded_models
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        workers=settings.WORKERS
    )
```

### 7. Docker Configuration

**`Dockerfile`**
```dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY client_examples/ ./client_examples/

# Create necessary directories
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/logs

# Set environment variables for Hugging Face
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**`docker-compose.yml` (CPU Version)**
```yaml
version: '3.8'

services:
  huggingface-multi-api:
    build: .
    container_name: huggingface-multi-api
    ports:
      - "8000:8000"
    environment:
      # Model Configuration
      - DEFAULT_MODEL=distilgpt2
      - MAX_CONCURRENT_MODELS=3
      - AUTO_UNLOAD_MINUTES=30
      
      # API Configuration
      - API_KEY=${API_KEY:-}
      - CORS_ORIGINS=*
      - LOG_LEVEL=INFO
      
      # Hugging Face Cache
      - HF_HOME=/app/.cache/huggingface
      - TRANSFORMERS_CACHE=/app/.cache/huggingface
      
      # Performance
      - WORKERS=1
      
    volumes:
      # Model cache (persistent)
      - ./models_cache:/app/.cache/huggingface
      # Logs
      - ./logs:/app/logs
      # Configuration (for easy editing)
      - ./app/config:/app/app/config
      
    restart: unless-stopped
    
    deploy:
      resources:
        limits:
          memory: 6G
        reservations:
          memory: 2G

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - huggingface-multi-api
    restart: unless-stopped
    profiles:
      - with-proxy

networks:
  default:
    name: huggingface-api-network
```

**`docker-compose.gpu.yml` (GPU Version)**
```yaml
version: '3.8'

services:
  huggingface-multi-api:
    build: .
    container_name: huggingface-multi-api-gpu
    ports:
      - "8000:8000"
    environment:
      # Model Configuration
      - DEFAULT_MODEL=distilgpt2
      - MAX_CONCURRENT_MODELS=2  # Fewer for GPU memory
      - AUTO_UNLOAD_MINUTES=30
      
      # API Configuration
      - API_KEY=${API_KEY:-}
      - CORS_ORIGINS=*
      - LOG_LEVEL=INFO
      
      # Hugging Face Cache
      - HF_HOME=/app/.cache/huggingface
      - TRANSFORMERS_CACHE=/app/.cache/huggingface
      
      # GPU Configuration
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      
      # Performance
      - WORKERS=1
      
    volumes:
      # Model cache (persistent)
      - ./models_cache:/app/.cache/huggingface
      # Logs
      - ./logs:/app/logs
      # Configuration (for easy editing)
      - ./app/config:/app/app/config
      
    restart: unless-stopped
    
    # GPU Runtime
    runtime: nvidia
    
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
      
    # GPU reservations (Docker Compose 3.8+)
    device_requests:
      - driver: nvidia
        count: 1
        capabilities: [gpu]

networks:
  default:
    name: huggingface-api-gpu-network
```

**`Dockerfile.gpu` (GPU-optimized version)**
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    python3.10-dev \
    git \
    curl \
    build-essential \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install GPU-enabled PyTorch
COPY requirements.gpu.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.gpu.txt

# Copy application code
COPY app/ ./app/
COPY client_examples/ ./client_examples/

# Create necessary directories
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/logs

# Set environment variables for Hugging Face
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**`requirements.gpu.txt`**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.35.2
torch==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
torchvision==0.16.1+cu118 --index-url https://download.pytorch.org/whl/cu118
tokenizers==0.15.0
accelerate==0.25.0
pydantic==2.5.0
python-multipart==0.0.6
PyYAML==6.0.1
numpy==1.24.3
safetensors==0.4.1
huggingface-hub==0.19.4
```

**`.dockerignore`**
```
.git
.gitignore
README.md
.pytest_cache
.coverage
.mypy_cache
**/__pycache__
**/*.pyc
models_cache/
logs/
*.log
.env
.venv
venv/
```

**`.gitignore`**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Models and cache
models_cache/
*.bin
*.safetensors

# Logs
logs/
*.log

# Docker
.dockerignore

# OS
.DS_Store
Thumbs.db
```

### 8. Client Examples

**`client_examples/basic_client.py`**
```python
import requests
import json
from typing import Dict, List, Optional

class HuggingFaceAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict:
        """List all available models"""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a specific model"""
        response = self.session.get(f"{self.base_url}/models/{model_id}/info")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_id: str, force_reload: bool = False) -> Dict:
        """Load a specific model"""
        payload = {"force_reload": force_reload}
        response = self.session.post(f"{self.base_url}/models/{model_id}/load", json=payload)
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_id: str) -> Dict:
        """Unload a specific model"""
        response = self.session.post(f"{self.base_url}/models/{model_id}/unload")
        response.raise_for_status()
        return response.json()
    
    def generate_text(
        self, 
        model_id: str, 
        text: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> Dict:
        """Generate text with a specific model"""
        payload = {
            "text": text,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences
        }
        response = self.session.post(f"{self.base_url}/models/{model_id}/generate", json=payload)
        response.raise_for_status()
        return response.json()
    
    def chat(
        self, 
        model_id: str, 
        message: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Optional[str] = None
    ) -> Dict:
        """Chat with a specific model"""
        payload = {"message": message}
        if max_length:
            payload["max_length"] = max_length
        if temperature:
            payload["temperature"] = temperature
        if context:
            payload["context"] = context
            
        response = self.session.post(f"{self.base_url}/models/{model_id}/chat", json=payload)
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = HuggingFaceAPIClient()
    
    try:
        # Check health
        health = client.health_check()
        print("✅ API Health:", health["status"])
        print(f"📊 Loaded models: {len(health['loaded_models'])}")
        
        # List all available models
        models = client.list_models()
        print(f"\n📚 Available models: {len(models)}")
        
        # Show some model options
        for model_id, info in list(models.items())[:5]:
            status = "🟢 LOADED" if info["loaded"] else "⚪ AVAILABLE"
            print(f"  {status} {model_id}: {info['description']} ({info['size']})")
        
        # Load and test a small model
        print(f"\n🔄 Loading distilgpt2...")
        load_result = client.load_model("distilgpt2")
        print(f"✅ {load_result['message']}")
        
        # Generate text
        print(f"\n🤖 Generating text...")
        result = client.generate_text(
            "distilgpt2",
            "The future of artificial intelligence is",
            max_length=80,
            temperature=0.8
        )
        print(f"📝 Generated: {result['generated_text'][0]}")
        
        # Test chat if conversational model is available
        conv_models = [mid for mid, info in models.items() if info['type'] == 'conversational']
        if conv_models:
            chat_model = conv_models[0]
            print(f"\n💬 Loading chat model: {chat_model}")
            client.load_model(chat_model)
            
            chat_result = client.chat(
                chat_model,
                "Hello! How are you today?",
                max_length=60
            )
            print(f"🗨️  Chat response: {chat_result['response']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
```

**`client_examples/multi_model_client.py`**
```python
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional

class AsyncMultiModelClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def compare_models(self, model_ids: List[str], prompt: str, **kwargs) -> Dict:
        """Compare multiple models on the same prompt"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Load all models first
            load_tasks = [self._load_model(session, model_id) for model_id in model_ids]
            await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Generate with all models simultaneously
            generate_tasks = [
                self._generate_text(session, model_id, prompt, **kwargs) 
                for model_id in model_ids
            ]
            results = await asyncio.gather(*generate_tasks, return_exceptions=True)
            
            # Organize results
            comparison = {}
            for model_id, result in zip(model_ids, results):
                if isinstance(result, Exception):
                    comparison[model_id] = {"error": str(result)}
                else:
                    comparison[model_id] = result
            
            return comparison
    
    async def _load_model(self, session: aiohttp.ClientSession, model_id: str):
        """Load a model"""
        async with session.post(f"{self.base_url}/models/{model_id}/load") as response:
            return await response.json()
    
    async def _generate_text(self, session: aiohttp.ClientSession, model_id: str, text: str, **kwargs):
        """Generate text with a specific model"""
        payload = {"text": text, **kwargs}
        async with session.post(f"{self.base_url}/models/{model_id}/generate", json=payload) as response:
            return await response.json()
    
    async def batch_generation(self, model_id: str, prompts: List[str], **kwargs) -> List[Dict]:
        """Generate text for multiple prompts with the same model"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Load model first
            await self._load_model(session, model_id)
            
            # Generate for all prompts
            tasks = [
                self._generate_text(session, model_id, prompt, **kwargs) 
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

# Example usage
async def main():
    client = AsyncMultiModelClient()
    
    # Compare multiple models
    print("🔄 Comparing models...")
    comparison = await client.compare_models(
        model_ids=["distilgpt2", "gpt2-small"],
        prompt="The benefits of renewable energy include",
        max_length=60,
        temperature=0.7
    )
    
    print("\n📊 Model Comparison Results:")
    for model_id, result in comparison.items():
        if "error" in result:
            print(f"❌ {model_id}: {result['error']}")
        else:
            print(f"✅ {model_id}: {result['generated_text'][0][:100]}...")
    
    # Batch generation
    print(f"\n🔄 Batch generation with distilgpt2...")
    prompts = [
        "The future of technology is",
        "Climate change requires",
        "Artificial intelligence will"
    ]
    
    batch_results = await client.batch_generation(
        "distilgpt2", 
        prompts, 
        max_length=50,
        temperature=0.8
    )
    
    print("\n📝 Batch Results:")
    for prompt, result in zip(prompts, batch_results):
        print(f"Prompt: {prompt}")
        print(f"Result: {result['generated_text'][0]}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

### 9. Setup and Deployment Instructions

**Step 1: Create Project Structure**
```bash
mkdir huggingface-multi-model-api
cd huggingface-multi-model-api

# Create directories
mkdir -p app/config client_examples models_cache logs

# Create __init__.py files
touch app/__init__.py app/config/__init__.py
```

**Step 2: Add All Files**
Copy all the files above into their respective locations in your project structure.

**Step 3: Choose Your Deployment Method**

### Option A: CPU Deployment (Default)
```bash
# Build and start with CPU
docker-compose build

docker-compose up -d

# With a HF token  - RUN THIS COMMAND IF YOU ARE USING GATED MODELS THAT REQUIRE A HF TOKEN. THIS WILL ALLOW THE BUILD TO USE THE HF TOKEN FOR GAINING ACESSS TO THE MODEL.
HF_TOKEN=actual_token_here docker-compose up -d

# Check logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### Option B: GPU Deployment

**Prerequisites for GPU:**
```bash
# Install NVIDIA Docker runtime
# Ubuntu/Debian:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

**Deploy with GPU:**
```bash
# Build GPU version
docker-compose -f docker-compose.gpu.yml build

# Start with GPU
docker-compose -f docker-compose.gpu.yml up -d

# Check GPU usage
docker exec -it huggingface-multi-api-gpu nvidia-smi

# Check logs
docker-compose -f docker-compose.gpu.yml logs -f
```

### Option C: Custom GPU Setup (Alternative)
If you prefer to modify the main docker-compose.yml:

```bash
# Edit docker-compose.yml and uncomment:
# runtime: nvidia
# - NVIDIA_VISIBLE_DEVICES=all

# Or use environment override
NVIDIA_VISIBLE_DEVICES=all docker-compose up -d
```

**Step 4: Test the API**
```bash
# List available models
curl http://localhost:8000/models | jq

# Load a model
curl -X POST http://localhost:8000/models/distilgpt2/load

# Generate text
curl -X POST http://localhost:8000/models/distilgpt2/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "max_length": 50}' | jq

# Chat with conversational model
curl -X POST http://localhost:8000/models/dialogpt-small/load
curl -X POST http://localhost:8000/models/dialogpt-small/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How are you?", "max_length": 50}' | jq
```

**Step 5: Use Python Clients**
```bash
# Test basic client
python client_examples/basic_client.py

# Test async multi-model client
python client_examples/multi_model_client.py
```

### GPU Troubleshooting

**Common GPU Issues and Solutions:**

1. **"runtime nvidia not found"**
   ```bash
   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **"no CUDA-capable device"**
   ```bash
   # Check if NVIDIA drivers are installed
   nvidia-smi
   
   # Check Docker can see GPU
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

3. **Out of GPU memory**
   ```bash
   # Reduce concurrent models in docker-compose.gpu.yml
   - MAX_CONCURRENT_MODELS=1
   
   # Or use smaller models in models.yaml
   ```

4. **CUDA version mismatch**
   ```bash
   # Check your CUDA version
   nvidia-smi
   
   # Update requirements.gpu.txt to match your CUDA version
   # For CUDA 11.8: torch==2.1.1+cu118
   # For CUDA 12.1: torch==2.1.1+cu121
   ```

### Performance Optimization

**For CPU Deployment:**
- Use smaller models (distilgpt2, opt-125m)
- Reduce MAX_CONCURRENT_MODELS to 2
- Increase memory allocation if needed

**For GPU Deployment:**
- Monitor GPU memory with `nvidia-smi`
- Use fp16 models for better memory efficiency
- Consider model quantization for larger models

### Environment Variables Reference

```bash
# Model Management
DEFAULT_MODEL=distilgpt2              # Auto-load this model at startup
MAX_CONCURRENT_MODELS=3               # Max models in memory
AUTO_UNLOAD_MINUTES=30               # Auto-unload after inactivity

# Security
API_KEY=your-secret-key              # Optional API authentication

# Performance  
WORKERS=1                            # Number of worker processes
LOG_LEVEL=INFO                       # Logging level

# GPU Settings (for GPU deployment)
NVIDIA_VISIBLE_DEVICES=all           # Which GPUs to use
CUDA_VISIBLE_DEVICES=0               # Specific GPU device
```

## How to Add New Models (Zero Code Changes)

One of the key features of this implementation is the ability to add new models without any code changes or rebuilding Docker images. Here's how to do it:

### Step-by-Step: Adding a New Model

#### Step 1: Add Model to Configuration

Edit `app/config/models.yaml` and add your new model:

```yaml
# Add this to the models section
your-new-model:
  name: "microsoft/DialoGPT-large"
  type: "conversational"
  size: "762M"
  description: "Large conversational model with better responses"
  max_length: 1000
  suggested_temperature: 0.7
```

#### Step 2: Restart the Service

```bash
# For CPU deployment
docker-compose restart

# For GPU deployment  
docker-compose -f docker-compose.gpu.yml restart

# Or reload config without restart (if service is running)
curl -X POST http://localhost:8000/admin/reload-config
```

#### Step 3: Verify Model is Available

```bash
# Check if your model appears in the list
curl http://localhost:8000/models | jq '.["your-new-model"]'

# Should return something like:
# {
#   "name": "microsoft/DialoGPT-large",
#   "type": "conversational",
#   "description": "Large conversational model with better responses",
#   "loaded": false,
#   "size": "762M"
# }
```

#### Step 4: Load and Test the Model

```bash
# Load the model (downloads automatically on first use)
curl -X POST http://localhost:8000/models/your-new-model/load

# Test text generation
curl -X POST http://localhost:8000/models/your-new-model/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how can I help you?", 
    "max_length": 100
  }' | jq

# Test chat functionality (for conversational models)
curl -X POST http://localhost:8000/models/your-new-model/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "max_length": 150
  }' | jq
```

#### Step 5: Use in Python Client

```python
from client_examples.basic_client import HuggingFaceAPIClient

client = HuggingFaceAPIClient()

# Load your new model
client.load_model("your-new-model")

# Generate text
result = client.generate_text(
    "your-new-model",
    "Explain quantum computing in simple terms",
    max_length=200,
    temperature=0.7
)

print(result["generated_text"][0])
```

### Examples: Different Model Types

#### Adding a Code Generation Model

```yaml
codegen-6b:
  name: "Salesforce/codegen-6B-mono"
  type: "code"
  size: "6B"
  description: "Large code generation model for programming tasks"
  max_length: 2048
  suggested_temperature: 0.2
```

**Usage:**
```bash
curl -X POST http://localhost:8000/models/codegen-6b/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "# Function to calculate fibonacci numbers\ndef fibonacci(n):",
    "max_length": 150,
    "temperature": 0.2
  }'
```

#### Adding an Instruction-Following Model

```yaml
flan-t5-large:
  name: "google/flan-t5-large"
  type: "instruction"
  size: "780M"
  description: "Large instruction-following model"
  max_length: 512
  suggested_temperature: 0.3
```

**Usage:**
```bash
curl -X POST http://localhost:8000/models/flan-t5-large/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Translate English to French: Hello, how are you?",
    "max_length": 100
  }'
```

#### Adding a Specialized Text Model

```yaml
gpt-neo-1.3b:
  name: "EleutherAI/gpt-neo-1.3B"
  type: "text"
  size: "1.3B"
  description: "GPT-Neo model for general text generation"
  max_length: 2048
  suggested_temperature: 0.8
```

### Model Configuration Reference

When adding a new model, here are the required and optional fields:

#### Required Fields:
- **`name`**: Hugging Face model identifier (e.g., "microsoft/DialoGPT-small")
- **`type`**: Model type - affects how the API handles the model
  - `"conversational"` - Chat/dialogue models
  - `"text"` - General text generation
  - `"code"` - Code generation models  
  - `"instruction"` - Instruction-following models
- **`size`**: Human-readable size (e.g., "117M", "1.3B")
- **`description`**: Brief description of the model
- **`max_length`**: Maximum sequence length for generation

#### Optional Fields:
- **`suggested_temperature`**: Recommended temperature for this model (default: 0.7)

### Model Types and Their Behavior

#### Conversational Models (`type: "conversational"`)
- **Endpoints**: `/generate`, `/chat`
- **Behavior**: Chat endpoint formats messages as "Human: ... Assistant: ..."
- **Examples**: DialoGPT models, BlenderBot

#### Text Generation Models (`type: "text"`)
- **Endpoints**: `/generate`
- **Behavior**: Direct text completion
- **Examples**: GPT-2, GPT-Neo, OPT

#### Code Models (`type: "code"`)
- **Endpoints**: `/generate`
- **Behavior**: Optimized for code completion
- **Examples**: CodeGen, CodeT5

#### Instruction Models (`type: "instruction"`)
- **Endpoints**: `/generate`
- **Behavior**: Follows instructions/prompts
- **Examples**: Flan-T5, T0

### Automatic Endpoint Generation

When you add a model with ID `my-model`, these endpoints are automatically created:

- **`GET /models/my-model/info`** - Model information
- **`POST /models/my-model/load`** - Load model into memory
- **`POST /models/my-model/unload`** - Unload model from memory
- **`POST /models/my-model/generate`** - Generate text
- **`POST /models/my-model/chat`** - Chat interface (all types support this)

### Tips for Adding Models

#### 1. **Check Model Compatibility**
Most Hugging Face models work, but ensure they're compatible with:
- `transformers` library
- AutoModelForCausalLM or AutoModelForSeq2SeqLM

#### 2. **Consider Memory Requirements**
- Small models: < 1GB RAM
- Medium models: 2-4GB RAM  
- Large models: 6GB+ RAM
- Adjust `MAX_CONCURRENT_MODELS` accordingly

#### 3. **Test First**
```bash
# Quick test to see if model works
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'your/model-name'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print('Model loaded successfully!')
"
```

#### 4. **GPU Memory Management**
For GPU deployment, monitor memory usage:
```bash
# Check GPU memory
docker exec -it huggingface-multi-api-gpu nvidia-smi

# If running out of memory, reduce concurrent models:
# Edit docker-compose.gpu.yml:
# - MAX_CONCURRENT_MODELS=1
```

### Removing Models

To remove a model:

1. **Remove from config:**
   ```bash
   # Delete the model section from app/config/models.yaml
   ```

2. **Reload config:**
   ```bash
   curl -X POST http://localhost:8000/admin/reload-config
   ```

3. **Or restart service:**
   ```bash
   docker-compose restart
   ```

The model will be automatically unloaded and its endpoints will no longer be available.

### Real-World Example: Adding ChatGPT-style Model

```yaml
# Add to models.yaml
vicuna-7b:
  name: "lmsys/vicuna-7b-v1.5"
  type: "conversational"
  size: "7B"
  description: "Vicuna chat model fine-tuned for conversations"
  max_length: 2048
  suggested_temperature: 0.7
```

```bash
# Restart and test
docker-compose restart

# Load model (will download ~13GB)
curl -X POST http://localhost:8000/models/vicuna-7b/load

# Chat with it
curl -X POST http://localhost:8000/models/vicuna-7b/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain the benefits of renewable energy",
    "max_length": 300
  }'
```

This process takes **less than 5 minutes** and requires **zero code changes**!