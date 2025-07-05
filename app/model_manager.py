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