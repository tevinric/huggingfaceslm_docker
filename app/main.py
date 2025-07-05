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