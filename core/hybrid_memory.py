from typing import Dict, Any, List
from .base import BaseMemory
from .external_memory import VectorMemory
from .basic_memory import SlidingWindowMemory

class HierarchicalMemory(BaseMemory):
    """
    Strategy 7: Hierarchical Memory (The Sandwich)
    
    Theory:
        Combines the best of both worlds: 
        1. Long-term Vector Search (to recall old facts).
        2. Short-term Sliding Window (to keep the conversation flowing).
    
    Pros:
        - Balanced approach.
        - Most production-grade systems use this (e.g., LangChain's default behavior).
    
    Cons:
        - Still loses "mid-term" context that isn't semantically similar but temporally relevant.
    """
    
    def __init__(self):
        super().__init__()
        self.long_term = VectorMemory()
        self.short_term = SlidingWindowMemory(window_size=3)
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Add to both layers
        self.long_term.add_message(role, content)
        self.short_term.add_message(role, content)
        
    def get_context(self, query: str) -> Dict[str, Any]:
        # 1. Get Long Term Context
        lt_context = self.long_term.get_context(query)
        
        # 2. Get Short Term Context
        st_context = self.short_term.get_context(query)
        
        # Combine
        final_prompt = "--- Long Term Facts (Relevant) ---\n"
        # Strip the "Relevant Past Context:\n" header from vector memory output to avoid duplication
        final_prompt += lt_context['final_prompt'].replace("Relevant Past Context:\n", "")
        
        final_prompt += "\n--- Short Term History (Recent) ---\n"
        final_prompt += st_context['final_prompt']
        
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "Hierarchical (Hybrid)",
                "long_term_retrieval": lt_context['debug_info'],
                "short_term_buffer": st_context['debug_info']
            }
        }
        
    def clear(self):
        super().clear()
        self.long_term.clear()
        self.short_term.clear()


class OSMemory(BaseMemory):
    """
    Strategy 8: OS-Like Memory (MemGPT Lite)
    
    Theory:
        Mimics an Operating System's paging mechanism.
        - RAM (Context Window): Expensive, fast, limited size.
        - Disk (Vector/SQL): Cheap, slow, unlimited.
        
        Explicitly SWAPS messages between RAM and Disk.
        
    Pros:
        - Gives the LLM "Agency" over its own memory.
        - Can handle infinite contexts intelligently.
    
    Cons:
        - Very complex prompts needed to teach LLM how to use the tools.
        - High latency due to multiple steps (Think -> Swap -> Answer).
    """
    
    def __init__(self, ram_limit: int = 5):
        super().__init__()
        self.ram_limit = ram_limit
        self.ram = [] # List of messages
        self.disk = [] # Simulated Disk (List of messages)
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        
        # New item enters RAM
        self.ram.append({"role": role, "content": content})
        
        # If RAM full, page out oldest to Disk
        if len(self.ram) > self.ram_limit:
            paged_out = self.ram.pop(0)
            self.disk.append(paged_out)
            
    def get_context(self, query: str) -> Dict[str, Any]:
        # Simulation: "Swap" check
        # In a real MemGPT, the LLM decides this. 
        # Here, we simulate: If query mentions "old" or "recall", we swap from disk.
        
        swapped_msg = None
        if "recall" in query.lower() or "remember" in query.lower():
             # Search disk (Linear scan for demo)
             for i, msg in enumerate(self.disk):
                 if msg['content'] in query: # Very dumb match for demo
                     swapped_msg = self.disk.pop(i)
                     self.ram.append(swapped_msg)
                     # Maintain size
                     if len(self.ram) > self.ram_limit:
                         self.disk.append(self.ram.pop(0))
                     break
        
        final_prompt = f"--- RAM State ({len(self.ram)}/{self.ram_limit}) ---\n"
        for msg in self.ram:
            final_prompt += f"[{msg['role']}]: {msg['content']}\n"
            
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "OS-Like (MemGPT Lite)",
                "ram_usage": f"{len(self.ram)}/{self.ram_limit}",
                "disk_count": len(self.disk),
                "ram_content": [m['content'] for m in self.ram],
                "disk_content": [m['content'] for m in self.disk],
                "last_swap_action": f"Moved '{swapped_msg['content']}' to RAM" if swapped_msg else "None"
            }
        }

