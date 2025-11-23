# 🧠 LLM Memory Patterns Playground

An educational tool to explore and understand how different memory strategies manipulate the Context Window in Large Language Models.

## 🎯 Purpose

This is **NOT** just a chatbot; it's an **inspector for the AI's brain**. Each memory strategy demonstrates different approaches to managing conversation history, with transparent debug information showing exactly what's happening under the hood.

## 🚀 Features

### 8 Memory Strategies

1. **Full Memory** - Remember everything (token explosion demo)
2. **Sliding Window** - Short-term focus (goldfish effect)
3. **Relevance Filtering** - Semantic similarity-based retrieval
4. **Summary Memory** - Lossy compression with recursive summarization
5. **Vector Memory (RAG)** - Infinite long-term memory with ChromaDB
6. **Knowledge Graph** - Structured facts with LLM-based triple extraction
7. **Hierarchical Memory** - Industry-standard hybrid approach
8. **OS-Like Memory** - RAM/Disk paging simulation (MemGPT-lite)

### Educational Features

- **Internal Monologue Inspector**: See exactly what context is sent to the LLM
- **Debug Information**: Transparent display of memory state, dropped messages, retrieved facts
- **Structured Output**: Uses Pydantic for reliable knowledge extraction
- **Multi-language Support**: Works with English and Chinese

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** - Interactive UI
- **LangChain** - LLM orchestration
- **OpenAI GPT-4o-mini** - LLM backend
- **ChromaDB** - Vector storage
- **NetworkX** - Knowledge graph

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-memory-patterns.git
cd llm-memory-patterns
```

2. Create virtual environment:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 🎮 Usage

Run the Streamlit app:
```bash
streamlit run app/playground.py
```

Then open your browser to `http://localhost:8502`

## 📚 Learning Path

1. Start with **Full Memory** to see the baseline
2. Try **Sliding Window** to understand the forgetting problem
3. Explore **Vector Memory (RAG)** for semantic retrieval
4. Test **Knowledge Graph** for structured reasoning
5. Compare **Hierarchical Memory** as the production-ready solution

## 🎓 Educational Goals

- Understand the tradeoffs between different memory strategies
- See how context window manipulation affects LLM responses
- Learn when to use each strategy in production systems
- Explore the limitations of each approach

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new memory strategies
- Improve extraction algorithms
- Enhance visualization
- Add more debug information

## 📄 License

MIT License

## 🙏 Acknowledgments

Built as an educational tool to demystify LLM memory management.

