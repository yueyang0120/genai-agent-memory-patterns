# LLM Memory Patterns

An interactive playground to explore how different memory strategies work in LLM applications. Built for developers who want to understand the tradeoffs before implementing memory in production.

## What is this?

When building chatbots or AI agents, you need to decide how to manage conversation history. This project implements 8 common approaches, with full transparency into what gets sent to the LLM at each turn.

Think of it as a debugging tool that shows you exactly what's happening inside the "memory" component.

## The 8 Strategies

**Basic**
- **Full Memory** - Keep everything (until you hit token limits)
- **Sliding Window** - Only keep last N messages

**Smart Filtering**
- **Relevance Filter** - Use embeddings to find similar past messages
- **Summary Memory** - Compress old messages into summaries

**External Storage**
- **Vector Memory (RAG)** - Store in vector DB, retrieve by similarity
- **Knowledge Graph** - Extract facts as triples, query the graph

**Hybrid**
- **Hierarchical** - Combine vector search + recent messages (most production systems use this)
- **OS-Inspired** - Simulate RAM/disk with explicit paging

## Quick Start

```bash
# clone and setup
git clone https://github.com/yueyang0120/genai-agent-memory-patterns.git
cd genai-agent-memory-patterns
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# add your openai key
cp env.example .env
# edit .env with your OPENAI_API_KEY

# run
streamlit run app/playground.py
```

## How to Use

1. Pick a strategy from the sidebar
2. Chat with the bot
3. Click "Internal Monologue & Memory State" to see:
   - What context was sent to the LLM
   - Which messages were kept/dropped
   - Similarity scores, summaries, graph facts, etc.

## Strategy Comparison

| Strategy | Memory Growth | Best For | Downside |
|----------|--------------|----------|----------|
| Full Memory | Linear | Short chats | Expensive at scale |
| Sliding Window | Constant | Stateless tasks | Forgets early context |
| Relevance Filter | Linear | Topic jumps | Misses conversation flow |
| Summary | Logarithmic | Long conversations | Loses details |
| Vector (RAG) | Constant retrieval | Knowledge Q&A | No temporal awareness |
| Knowledge Graph | Depends on facts | Structured reasoning | Hard to extract reliably |
| Hierarchical | Constant | Production apps | More complex |
| OS-Inspired | Constant | Agent frameworks | Needs explicit swapping logic |

## Tech Stack

- Python 3.10+
- Streamlit (UI)
- LangChain (LLM wrapper)
- OpenAI GPT-4o-mini
- ChromaDB (vector store)
- NetworkX (graph)

## Architecture Notes

### Base Class
All strategies inherit from `BaseMemory`:
```python
def add_message(role: str, content: str)
def get_context(query: str) -> dict  # returns {final_prompt, debug_info}
def clear()
```

### Knowledge Graph
Uses GPT-4o-mini with structured output to extract triples:
```python
"I work at Google" → (user, work_at, google)
```

### Vector Memory
Stores each message as an embedding in ChromaDB, retrieves top-k by similarity.

### Hierarchical
Combines two layers:
- Long-term: Vector search (top 2)
- Short-term: Sliding window (last 3)

## When to Use What

**Use Full Memory if:**
- Conversations are short (<10 turns)
- You need perfect recall

**Use Sliding Window if:**
- Each message is independent
- You only care about recent context

**Use Vector Memory if:**
- Users ask about things mentioned long ago
- You have a knowledge base to search

**Use Hierarchical if:**
- Building a production chatbot
- Need balance between recency and relevance

## Implementation Details

The Knowledge Graph uses Pydantic structured output for reliable extraction (no regex, no eval):
```python
class Triple(BaseModel):
    subject: str
    relation: str
    object: str
```

Entity extraction for graph queries also uses structured output to avoid hardcoded patterns.

## Contributing

PRs welcome. Ideas:
- Add more strategies (e.g., attention-based, learned compression)
- Better visualization
- Benchmark suite
- Multi-turn evaluation metrics

## License

MIT

## Why I Built This

Most tutorials show you *how* to add memory to an LLM app, but don't explain *which* strategy to use. This project lets you see the tradeoffs firsthand by inspecting the actual context sent to the model.
