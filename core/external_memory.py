from __future__ import annotations
from typing import Dict, Any, List
import os
import time
import networkx as nx
from .base import BaseMemory
from utils.text_utils import llm_triples_extract
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Conditional import for ChromaDB to handle version compatibility
try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except Exception as e:
    CHROMA_AVAILABLE = False
    CHROMA_ERROR = str(e)

class VectorMemory(BaseMemory):
    """
    Strategy 5: Vector Memory (Infinite Long-Term Memory / RAG)
    
    Theory:
        Offloads memory to a dedicated Vector Database (ChromaDB).
        Retrieves the top-k most relevant fragments based on semantic similarity.
        Ignores time order; focuses on content match.
    
    Pros:
        - True "Long Term Memory" (can recall facts from months ago).
        - Scalable to millions of messages.
    
    Cons:
        - Complexity (requires external DB).
        - Retrieved fragments might lack context (Who said what? When?).
        
    Best Use Case:
        - Knowledge Base bots.
        - Personal Assistants that need to remember user preferences over long periods.
    """
    
    def __init__(self):
        super().__init__()
        self.error_msg = None
        
        if not CHROMA_AVAILABLE:
            self.vector_store = None
            self.error_msg = f"ChromaDB not available (Python 3.9+ required): {CHROMA_ERROR}"
            print(f"ERROR: {self.error_msg}")
            return
            
        try:
            self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
            # Using an ephemeral in-memory client for the playground
            self.vector_store = Chroma(
                collection_name="chat_history",
                embedding_function=self.embeddings,
                persist_directory="./chroma_db_ephemeral" # For demo purposes
            )
        except Exception as e:
            self.vector_store = None
            self.error_msg = f"VectorMemory initialization failed: {str(e)}"
            print(f"ERROR: {self.error_msg}")

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if self.vector_store:
            doc = Document(
                page_content=content,
                metadata={
                    "role": role, 
                    "timestamp": time.time(),
                    "message_index": len(self.history) - 1
                }
            )
            self.vector_store.add_documents([doc])

    def get_context(self, query: str) -> Dict[str, Any]:
        if not self.vector_store:
             return {
                 "final_prompt": "Error: Vector Store not initialized", 
                 "debug_info": {
                     "strategy": "Vector Memory (RAG)",
                     "error": self.error_msg or "Unknown initialization error"
                 }
             }
             
        # Retrieve top 5 similar messages with scores
        results = self.vector_store.similarity_search_with_score(query, k=5)
        
        final_prompt = "Relevant Past Context:\n"
        retrieved_debug = []
        
        for doc, score in results:
            role = doc.metadata.get("role", "unknown")
            msg_idx = doc.metadata.get("message_index", "?")
            final_prompt += f"[{role}]: {doc.page_content}\n"
            retrieved_debug.append({
                "content": f"[{role}] {doc.page_content[:100]}...",  # Truncate for readability
                "similarity_score": f"{score:.4f}",
                "message_index": msg_idx
            })
            
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "Vector Memory (RAG)",
                "query": query,
                "retrieved_count": len(results),
                "retrieved_fragments": retrieved_debug,
                "total_docs_in_db": len(self.history),
                "note": "Lower score = more similar. RAG retrieves by semantic similarity, not chronological order."
            }
        }
        
    def clear(self):
        super().clear()
        if self.vector_store:
             self.vector_store.delete_collection()
             # Re-init to create empty collection
             self.vector_store = Chroma(
                collection_name="chat_history",
                embedding_function=self.embeddings,
                 persist_directory="./chroma_db_ephemeral"
            )


class GraphMemory(BaseMemory):
    """
    Strategy 6: Knowledge Graph (Structured Facts)
    
    Theory:
        Extracts structured facts (Triples) from messages and builds a Graph.
        Retrieves neighbors of entities found in the query.
    
    Pros:
        - Explicit reasoning chains ("A is B, B is C -> A is C").
        - High precision for factual data.
    
    Cons:
        - Very hard to implement "Triple Extraction" reliably (LLMs are inconsistent).
        - Graph traversal logic can get complex.
        
    Best Use Case:
        - Detective games, Scientific research assistants.
    """
    
    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        
        # Extract triples using LLM with structured output
        triples = llm_triples_extract(content)
            
        for subj, pred, obj in triples:
            self.graph.add_edge(subj, obj, relation=pred)

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Use LLM to extract entities from the query."""
        try:
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field
            
            class QueryEntities(BaseModel):
                entities: List[str] = Field(description="List of entities mentioned in the query (lowercase)")
            
            llm = ChatOpenAI(
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini"
            ).with_structured_output(QueryEntities)
            
            prompt = f"""Extract all entities (people, places, organizations, concepts) from this query.
Return them in lowercase.

Query: "{query}"
"""
            result = llm.invoke(prompt)
            return result.entities
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            # Fallback to simple word splitting
            return [w.lower() for w in query.split() if len(w) > 2]
    
    def get_context(self, query: str) -> Dict[str, Any]:
        # 1. Extract entities from query using LLM
        query_entities = self._extract_entities_from_query(query)
        relevant_facts = []
        visited_nodes = set()
        
        # 2. Find matching nodes in graph
        for entity in query_entities:
            if entity in self.graph.nodes:
                visited_nodes.add(entity)
                
                # Get outgoing edges (facts where entity is Subject)
                for neighbor in self.graph.neighbors(entity):
                    relation = self.graph[entity][neighbor]['relation']
                    relevant_facts.append(f"{entity} {relation} {neighbor}")
                
                # Get incoming edges (facts where entity is Object)
                for predecessor in self.graph.predecessors(entity):
                    relation = self.graph[predecessor][entity]['relation']
                    relevant_facts.append(f"{predecessor} {relation} {entity}")
        
        # 3. If no facts found, get all facts for LLM to filter
        if not relevant_facts:
            for u, v, data in self.graph.edges(data=True):
                relevant_facts.append(f"{u} {data['relation']} {v}")
                    
        final_prompt = "Known Facts (Graph):\n"
        if relevant_facts:
            final_prompt += "\n".join(set(relevant_facts))
        else:
            final_prompt += "No relevant facts found."
        
        # Get all edges for display
        all_facts = []
        for u, v, data in self.graph.edges(data=True):
            all_facts.append(f"{u} {data['relation']} {v}")
        
        return {
            "final_prompt": final_prompt,
            "debug_info": {
                "strategy": "Knowledge Graph (LLM-based Extraction & Retrieval)",
                "query": query,
                "extracted_entities": query_entities,
                "nodes_found_in_graph": list(visited_nodes),
                "facts_retrieved": list(set(relevant_facts)),
                "retrieval_method": "Entity-based" if visited_nodes else "Full graph (for LLM filtering)",
                "total_graph_nodes": self.graph.number_of_nodes(),
                "total_graph_edges": self.graph.number_of_edges(),
                "all_facts_in_graph": all_facts[:15]  # Show first 15 for readability
            }
        }

