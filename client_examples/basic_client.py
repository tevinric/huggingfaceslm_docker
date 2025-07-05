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