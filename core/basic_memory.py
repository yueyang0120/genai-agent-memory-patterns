from typing import Dict, Any
from collections import deque
from .base import BaseMemory

class FullMemory(BaseMemory):
    """
    Strategy 1: Full Memory (Remember Everything)
    
    Theory:
        Stores every new message in a simple list.
        The context window grows linearly with the conversation.
    
    Pros:
        - Perfect recall of the entire conversation.
        - Simplest implementation.
    
    Cons:
        - Token count explodes (Costly $$).
        - Will eventually hit the LLM's context limit.
        - "Lost in the Middle" phenomenon (LLMs pay less attention to the middle of long prompts).
        
    Best Use Case:
        - Short, finite conversations.
        - Debugging.
    """
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        
    def get_context(self, query: str) -> Dict[str, Any]:
        # In a real app, you'd format this as a chat string
        # Here we return the raw list for demonstration
        final_prompt = ""
        for msg in self.history:
            final_prompt += f"{msg['role']}: {msg['content']}\n"
            
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "Full Memory",
                "history_length": len(self.history),
                "total_tokens_approx": len(final_prompt.split()) * 1.3, # Rough estimate
                "full_history": self.history
            }
        }


class SlidingWindowMemory(BaseMemory):
    """
    Strategy 2: Sliding Window (Short-term Focus)
    
    Theory:
        Maintains a fixed-size buffer (deque).
        When adding a message, if len > k, the oldest message is dropped.
        Simulates "Goldfish Memory".
    
    Pros:
        - Constant token usage (Predictable Cost).
        - Never hits context limit (if window is small enough).
    
    Cons:
        - Catastrophic forgetting: Critical early details are lost.
        - Can lose context if the user refers to something 6 messages ago.
        
    Best Use Case:
        - Chatbots where only the immediate context matters (e.g., customer support turn-taking).
    """
    
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
        # Use a deque for efficient popping from the left
        self.window = deque(maxlen=window_size)
        
    def add_message(self, role: str, content: str):
        # We store in main history for record-keeping, but the logic uses the window
        self.history.append({"role": role, "content": content})
        self.window.append({"role": role, "content": content})
        
    def get_context(self, query: str) -> Dict[str, Any]:
        final_prompt = ""
        for msg in self.window:
            final_prompt += f"{msg['role']}: {msg['content']}\n"
            
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "Sliding Window",
                "window_size": self.window_size,
                "current_buffer_length": len(self.window),
                "dropped_messages_count": max(0, len(self.history) - len(self.window)),
                "current_buffer": list(self.window)
            }
        }

