from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseMemory(ABC):
    """
    Abstract Base Class for all Memory Strategies.
    
    This class defines the contract that all memory implementations must follow.
    It ensures consistency across different strategies for the Streamlit playground.
    """
    
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        
    @abstractmethod
    def add_message(self, role: str, content: str):
        """
        Add a message to the memory state.
        
        Args:
            role (str): 'user' or 'assistant'
            content (str): The text content of the message
        """
        pass
        
    @abstractmethod
    def get_context(self, query: str) -> Dict[str, Any]:
        """
        Retrieve context based on the current memory state and the new user query.
        
        Args:
            query (str): The new user query (used for relevance search in some strategies)
            
        Returns:
            Dict containing:
                - 'final_prompt': The actual string to send to the LLM
                - 'debug_info': A dictionary of internal state for educational display
        """
        pass
        
    def clear(self):
        """Reset memory state."""
        self.history = []

