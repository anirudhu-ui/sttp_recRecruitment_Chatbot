# Attributions

RecruitFlow AI is original work built for the Vardhaman College of
Engineering Agentic AI Bootcamp Hackathon. It uses the following
open-source libraries and third-party APIs; none of their code is
copied into this repository, only their public packages are installed
as dependencies.

- **LangGraph / LangChain** - agent orchestration, graph state machine, checkpointing
- **ChromaDB** (via `langchain-chroma`) - vector store for resume RAG
- **Groq** (via `langchain-groq`, model `llama-3.3-70b-versatile`) - LLM inference
- **Google Generative AI Embeddings** (via `langchain-google-genai`, `models/gemini-embedding-001`) - resume embeddings
- **Tavily** (via `langchain-tavily`) - web search for salary research
- **Pydantic** - structured output schemas

We did not create any of these libraries or APIs; credit belongs to
their respective maintainers.
