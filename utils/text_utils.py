import numpy as np
from typing import List, Tuple
import re
import os
from pydantic import BaseModel, Field

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        v1 (List[float]): First vector
        v2 (List[float]): Second vector
        
    Returns:
        float: Similarity score between -1 and 1
    """
    # Convert to numpy arrays
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    
    # Compute dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

class Triple(BaseModel):
    """A single knowledge triple."""
    subject: str = Field(description="The subject entity (lowercase, no spaces)")
    relation: str = Field(description="The relationship/predicate (lowercase, use underscore for spaces)")
    object: str = Field(description="The object entity (lowercase, no spaces)")

class KnowledgeTriples(BaseModel):
    """Collection of extracted knowledge triples."""
    triples: List[Triple] = Field(description="List of extracted triples from the text")

def llm_triples_extract(text: str) -> List[Tuple[str, str, str]]:
    """
    LLM-based extraction of (Subject, Predicate, Object) triples using Structured Output.
    Uses GPT with Pydantic schema to reliably extract facts from natural language.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Tuple[str, str, str]]: List of triples
    """
    try:
        from langchain_openai import ChatOpenAI
        
        # Use structured output with Pydantic
        llm = ChatOpenAI(
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        ).with_structured_output(KnowledgeTriples)
        
        prompt = f"""Extract factual knowledge triples (Subject, Relation, Object) from the following text.

Guidelines:
- Convert entities to lowercase
- Use underscores for multi-word relations (e.g., "live_in", "work_at")
- Extract all meaningful facts
- For "I" or "my", use "user" as the subject

Examples:
- "I live in Singapore" -> (user, live_in, singapore)
- "Alice likes Python" -> (alice, likes, python)  
- "My name is Bob" -> (user, name_is, bob)
- "我住在新加坡" -> (user, live_in, singapore)

Text: "{text}"
"""

        result: KnowledgeTriples = llm.invoke(prompt)
        
        # Convert Pydantic models to tuples
        triples = [(t.subject, t.relation, t.object) for t in result.triples]
        
        return triples
            
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return []  # Return empty list if extraction fails

