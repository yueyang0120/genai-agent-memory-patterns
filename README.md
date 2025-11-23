# LLM Memory Patterns

A comparative study of context window management strategies for large language models, implemented as an interactive educational tool.

## Overview

This project provides a systematic exploration of eight distinct memory management approaches used in conversational AI systems. Each implementation includes transparent introspection capabilities to demonstrate how different strategies manipulate the context window and affect model behavior.

## Implemented Strategies

### Basic Approaches
1. **Full Memory** - Maintains complete conversation history without compression
2. **Sliding Window** - Fixed-size buffer with FIFO eviction policy

### Compression & Filtering
3. **Relevance Filtering** - Embedding-based semantic similarity retrieval
4. **Summary Memory** - Recursive summarization with buffer management

### External Storage
5. **Vector Memory (RAG)** - Persistent vector database with similarity search
6. **Knowledge Graph** - Structured fact extraction and graph-based retrieval

### Hybrid Systems
7. **Hierarchical Memory** - Multi-tier architecture combining vector search and recency
8. **OS-Inspired Memory** - Paging mechanism with explicit RAM/disk separation

## Key Features

- Transparent context inspection for each strategy
- Real-time visualization of memory state transitions
- Structured output using Pydantic schemas
- LLM-based entity extraction for knowledge graphs
- Comparative performance analysis across strategies

## Technical Stack

- **Python 3.10+**
- **Streamlit** - Interactive web interface
- **LangChain** - LLM orchestration framework
- **OpenAI GPT-4o-mini** - Language model backend
- **ChromaDB** - Vector storage and retrieval
- **NetworkX** - Graph data structure and algorithms

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yueyang0120/genai-agent-memory-patterns.git
cd genai-agent-memory-patterns
```

2. Create and activate virtual environment:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp env.example .env
# Add your OPENAI_API_KEY to .env
```

## Usage

Start the interactive playground:
```bash
streamlit run app/playground.py
```

Access the interface at `http://localhost:8502`

## Architecture

### Base Memory Class
All strategies inherit from `BaseMemory` abstract class with three core methods:
- `add_message(role, content)` - Append new conversation turn
- `get_context(query)` - Retrieve relevant context for current query
- `clear()` - Reset memory state

### Context Retrieval
Each `get_context()` call returns:
- `final_prompt` - Formatted context string for LLM
- `debug_info` - Internal state for visualization

## Strategy Comparison

| Strategy | Token Growth | Recall | Latency | Use Case |
|----------|-------------|--------|---------|----------|
| Full Memory | O(n) | Perfect | Low | Short conversations |
| Sliding Window | O(1) | Recent only | Low | Stateless interactions |
| Relevance Filter | O(n) | Semantic | Medium | Topic-focused dialogue |
| Summary | O(log n) | Lossy | High | Long-form conversations |
| Vector (RAG) | O(1) retrieval | Semantic | Medium | Knowledge-intensive tasks |
| Knowledge Graph | O(edges) | Structured | High | Factual reasoning |
| Hierarchical | O(k) | Balanced | Medium | Production systems |
| OS-Inspired | O(k) | Explicit | Medium | Agent frameworks |

## Educational Goals

- Understand tradeoffs between memory strategies
- Observe context window manipulation effects
- Learn when to apply each approach in production
- Explore limitations through transparent debugging

## Implementation Details

### Knowledge Graph
Uses structured output with Pydantic models for reliable triple extraction:
```python
class Triple(BaseModel):
    subject: str
    relation: str
    object: str
```

### Vector Memory
Implements similarity search with ChromaDB:
- Automatic embedding generation
- Top-k retrieval with distance scores
- Metadata filtering by timestamp and role

### Hierarchical Memory
Combines two layers:
- Long-term: Vector search (top-2 relevant facts)
- Short-term: Sliding window (last 3 messages)

## Contributing

Contributions are welcome. Areas for improvement:
- Additional memory strategies
- Enhanced retrieval algorithms
- Performance benchmarks
- Visualization improvements

## License

MIT License

## References

This project synthesizes concepts from:
- Retrieval-Augmented Generation (RAG)
- MemGPT architecture
- LangChain memory modules
- Context window optimization techniques
