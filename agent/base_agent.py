"""
Base Agent class for all multi-agent orchestration patterns.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import google.generativeai as genai
import asyncio
import time


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, role: str, api_key: str, model: str = "gemini-2.5-flash"):
        self.name = name
        self.role = role
        genai.configure(api_key=api_key)
        self.model = model
        self.execution_time: float = 0.0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Process input data and return output."""
        pass
    
    async def _make_api_call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Make an API call to Google Gemini and track metrics."""
        start_time = time.time()
        
        try:
            model = genai.GenerativeModel(self.model)
           
            system_prompt = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt += msg["content"] + "\n"
                elif msg["role"] == "user":
                    # Prepend system prompt to the first user message
                    content = system_prompt + msg["content"] if system_prompt else msg["content"]
                    chat_messages.append({'role': 'user', 'parts': [content]})
                    system_prompt = "" # Clear after use
                elif msg["role"] == "assistant":
                     chat_messages.append({'role': 'model', 'parts': [msg["content"]]})

            response = await model.generate_content_async(
                chat_messages,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            
            self.execution_time = time.time() - start_time
            
            return response.text
            
        except Exception as e:
            self.execution_time = time.time() - start_time
            raise Exception(f"API call failed for {self.name}: {str(e)}")
    
    def get_metrics(self) -> Dict:
        """Get execution metrics for this agent."""
        return {
            "name": self.name,
            "role": self.role,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage
        }


class SimpleAgent(BaseAgent):
    """A simple agent implementation for basic tasks."""
    
    def __init__(self, name: str, role: str, api_key: str, system_prompt: str, model: str = "gemini-2.5-flash"):
        super().__init__(name, role, api_key, model)
        self.system_prompt = system_prompt
    
    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Process input using the system prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(input_data)}
        ]
        
        if context:
            messages.insert(1, {"role": "system", "content": f"Context: {context}"})
        
        result = await self._make_api_call(messages)
        return result
