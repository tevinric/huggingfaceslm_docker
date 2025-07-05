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