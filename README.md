# Gen AI & Agentic AI

Small Python projects covering embeddings, vector search, retrieval-augmented generation (RAG), and LangGraph workflows.

## Project Structure

- `Day 1/` - tokenization, TF-IDF, one-hot encoding, ChromaDB, and portfolio matching.
- `Day 2/` - PDF-based RAG and a Streamlit career guidance chatbot.
- `Day 3/` - a Streamlit code analyzer built with LangGraph and a Groq-hosted LLM.

## Setup

Use Python 3.10 or newer and run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r "Day 1\requirements.txt"
python -m pip install langgraph streamlit chromadb PyPDF2 langchain-text-splitters
```

The dependency file is kept in `Day 1/` because it was created with the first set of exercises. Install any package versions required by your selected exercise before running it.

## API Key

Create a `.env` file in the repository root and add your Groq key:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit API keys. `.env` and common local secret files are ignored by Git. The current example scripts contain placeholder values; replace those with environment-variable loading before using a real key.

## Run Examples

Run standalone scripts with:

```powershell
python "Day 1\chromdb.py"
python "Day 2\RAG.py"
```

Run Streamlit applications with:

```powershell
streamlit run "Day 2\careerchatbot.py"
streamlit run "Day 3\codeanalyzerapp.py"
```

`Day 2\RAG.py` expects `Day 2\LLM.pdf` to be present. The Streamlit career chatbot accepts PDF files through its upload control.

## LangGraph Note

The Day 3 application imports `create_react_agent` from `langgraph.prebuilt`. This API is available in LangGraph but may appear struck through in VS Code because newer releases mark it as deprecated. The strike-through is a deprecation indicator, not an installation failure.