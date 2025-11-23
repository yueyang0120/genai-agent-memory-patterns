from typing import Dict, Any, List
import os
from .base import BaseMemory
from utils.text_utils import cosine_similarity
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

class RelevanceMemory(BaseMemory):
    """
    Strategy 3: Relevance Filtering (Only What Matters)
    
    Theory:
        Stores all history but only retrieves messages semantically similar to the query.
        Uses Vector Embeddings (OpenAI) + Cosine Similarity.
    
    Pros:
        - Reduces noise.
        - Can recall specific details from long ago without processing the whole history.
    
    Cons:
        - Computationally expensive (embedding every query).
        - "Context Fragmentation": Might miss the flow of conversation if messages are disjointed.
        
    Best Use Case:
        - Q&A bots over a specific session where the user jumps between topics.
    """
    
    def __init__(self, threshold: float = 0.75):
        super().__init__()
        self.threshold = threshold
        # We'll store embeddings alongside messages
        # Structure: {'role': str, 'content': str, 'embedding': List[float]}
        self.memory_bank = []
        self.error_msg = None
        try:
            self.embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
             self.embeddings_model = None
             self.error_msg = f"RelevanceMemory initialization failed: {str(e)}"
             print(f"ERROR: {self.error_msg}")

    def add_message(self, role: str, content: str):
        embedding = []
        if self.embeddings_model:
            try:
                embedding = self.embeddings_model.embed_query(content)
            except Exception as e:
                print(f"Embedding error: {e}")
        
        self.memory_bank.append({
            "role": role, 
            "content": content, 
            "embedding": embedding
        })
        self.history.append({"role": role, "content": content})

    def get_context(self, query: str) -> Dict[str, Any]:
        if not self.embeddings_model:
             return {
                 "final_prompt": "Error: No Embedding Model", 
                 "debug_info": {
                     "strategy": "Relevance Filtering",
                     "error": self.error_msg or "Embedding model not initialized"
                 }
             }

        query_embedding = self.embeddings_model.embed_query(query)
        
        relevant_messages = []
        dropped_messages = []
        
        for msg in self.memory_bank:
            if not msg['embedding']:
                continue
                
            score = cosine_similarity(query_embedding, msg['embedding'])
            
            if score >= self.threshold:
                relevant_messages.append({**msg, "score": score})
            else:
                dropped_messages.append({**msg, "score": score})
                
        # Sort relevant by chronological order (or relevance, but usually chronological makes more sense for reading)
        # Here we just keep original order
        
        final_prompt = ""
        for msg in relevant_messages:
            final_prompt += f"{msg['role']}: {msg['content']}\n"
            
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "Relevance Filtering",
                "threshold": self.threshold,
                "kept_count": len(relevant_messages),
                "dropped_count": len(dropped_messages),
                "kept_messages": [{"content": m['content'], "score": f"{m['score']:.4f}"} for m in relevant_messages],
                "dropped_messages": [{"content": m['content'], "score": f"{m['score']:.4f}"} for m in dropped_messages]
            }
        }

class SummaryMemory(BaseMemory):
    """
    Strategy 4: Summary Memory (Lossy Compression)
    
    Theory:
        Maintains a "Running Summary" and a "Short-term Buffer".
        When the buffer gets full, it triggers an LLM call to summarize the buffer 
        and merge it into the existing summary.
    
    Pros:
        -Infinite conversation length (in theory).
        - Compresses tokens significantly.
    
    Cons:
        - "Chinese Whispers" effect: Details get distorted over time.
        - Loss of verbatim quotes.
        - Latency spike when summarization triggers.
        
    Best Use Case:
        - Long RPG roleplay or therapeutic bots where the "gist" matters more than specific words.
    """
    
    def __init__(self, buffer_limit: int = 3):
        super().__init__()
        self.summary = ""
        self.buffer_limit = buffer_limit
        self.buffer = [] # List of formatted strings
        self.error_msg = None
        try:
            self.llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            self.llm = None
            self.error_msg = f"SummaryMemory LLM initialization failed: {str(e)}"
            print(f"ERROR: {self.error_msg}")
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.buffer.append(f"{role}: {content}")
        
        if len(self.buffer) > self.buffer_limit:
            self._update_summary()
            
    def _update_summary(self):
        if not self.llm:
            print("Cannot summarize: LLM not initialized")
            return
            
        # Create a conversation string from the buffer
        conversation_text = "\n".join(self.buffer)
        
        prompt = f"""
        Existing Summary: "{self.summary}"
        
        New Lines:
        {conversation_text}
        
        Task: Update the summary to include the new lines. Keep it concise. 
        If there was no existing summary, just summarize the new lines.
        """
        
        try:
            response = self.llm.invoke(prompt)
            self.summary = response.content
            self.buffer = [] # Clear buffer after summarization
        except Exception as e:
            print(f"Summarization failed: {e}")

    def get_context(self, query: str) -> Dict[str, Any]:
        if not self.llm:
            return {
                "final_prompt": "Error: LLM not initialized",
                "debug_info": {
                    "strategy": "Summary Memory",
                    "error": self.error_msg or "LLM not initialized"
                }
            }
        
        final_prompt = f"System: Here is a summary of the past conversation: {self.summary}\n\n"
        final_prompt += "Recent Conversation:\n"
        for line in self.buffer:
            final_prompt += f"{line}\n"
            
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "Summary Memory",
                "current_summary": self.summary,
                "buffer_length": len(self.buffer),
                "buffer_content": self.buffer
            }
        }

