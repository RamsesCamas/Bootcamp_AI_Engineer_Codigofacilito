# AI Engineer Bootcamp

Base project for the Código Facilito AI Engineer Bootcamp. Integrates with Gemini and Groq APIs for LLM experimentation, token analysis, streaming comparisons, prompt engineering evaluation, structured outputs, and RAG pipelines.

## Project Structure

```
bootcamp-ai-engineer-codigofacilito/
├── slides/                             # Presentation slides (PDFs/PPTX by class)
│   ├── Clase-01/                       # Welcome & bootcamp intro
│   ├── Clase-02/                       # Tokens, context, cost & latency
│   ├── Clase-03/                       # Prompt Engineering
│   ├── Clase-04/                       # Structured Outputs & Contracts
│   └── Clase-05/                       # RAG Foundations
└── ai-engineer-bootcamp/
    ├── main.py                         # CLI demo: chat with LLM, shows usage metrics
    ├── main_rag.py                     # Interactive RAG pipeline demo (Clase 5)
    ├── pyproject.toml                  # Project config & dependencies
    ├── requirements.txt                # Pip dependencies
    ├── .env.example                    # Template for environment variables
    ├── core/
    │   ├── config.py                   # Loads settings from .env
    │   ├── llm_client.py               # Gemini/Groq LLM client (chat, usage)
    │   ├── logger.py                   # Logging setup
    │   └── tokenlab.py                 # Token counting, latency, streaming, budget
    ├── scripts/
    │   ├── streamlit_tokenlab.py       # Streamlit app: streaming vs non-streaming
    │   └── practice_tokenlab.py        # CLI practice script (~30 min)
    ├── prompting/                      # Clase 3 — Prompt Engineering toolkit
    │   ├── promptkit.py                # PromptTemplate, PromptRegistry, PromptChain, evaluate_prompt
    │   └── templates/
    │       └── ticket_classifier.py    # Ticket classifier templates v1–v4 + chain
    ├── practice/                       # Clase 3 — Evaluation runners
    │   ├── clase3_runner.py            # Evaluates v1–v4 with Gemma + best version on Groq
    │   ├── clase3_runner_groq.py       # Groq-only runner with live per-ticket output
    │   └── golden_set_tickets.json     # Labeled dataset for prompt evaluation
    ├── structured_outputs/             # Clase 4 — Structured Outputs & Function Calling
    │   ├── 01_pydantic_basico.py       # Pydantic basics
    │   ├── 02_modelos_anidados.py      # Nested models
    │   ├── 03_json_mode.py             # JSON mode with LLMs
    │   ├── 04_structured_outputs.py    # Native structured outputs
    │   ├── 05_function_calling_definicion.py  # Function calling: definition
    │   ├── 06_function_calling_completo.py    # Function calling: full loop
    │   ├── 07_pydantic_json_schema.py  # Pydantic → JSON Schema
    │   ├── 08_practica_paso1_contrato.py      # Practice: data contract
    │   ├── 09_practica_paso2_extraccion.py    # Practice: extraction
    │   ├── 10_practica_paso3_produccion.py    # Practice: production hardening
    │   ├── 11_practica_integrada.py    # Full integrated practice
    │   ├── actividad_extractor_recetas.py     # Activity: recipe extractor
    │   └── Tarea.md                    # Homework assignments
    ├── rag/                            # Clase 5 — RAG pipeline modules
    │   ├── ingestion.py                # Document loaders (txt, pdf, md) + paragraph chunker
    │   ├── embeddings.py               # Sentence-transformer embeddings (multilingual MiniLM)
    │   └── vectorstore.py              # ChromaDB integration: index, upsert, cosine search
    └── data/                           # Sample documents for RAG (NovaTech corp docs)
        ├── langchain-readme.md
        ├── manual_onboarding.txt
        ├── politica_vacaciones.txt
        └── proceso_soporte_tecnico.txt
```

## Setup

### 1. Python

Requires **Python 3.10+** (see `pyproject.toml`).

### 2. Virtual Environment

```bash
cd ai-engineer-bootcamp
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### 4. Environment Variables

Copy the example file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set at least one provider:

**Gemini (default):**
- `GEMINI_API_KEY` — Get from [Google AI Studio](https://aistudio.google.com/apikey)
- `LLM_PROVIDER=gemini`
- `LLM_MODEL` — e.g. `gemini-3-flash-preview` or `gemma-3-27b-it`

**Groq (optional):**
- `GROQ_API_KEY` — Get from [Groq Console](https://console.groq.com/)
- `GROQ_MODEL` — e.g. `llama-3.3-70b-versatile` or `openai/gpt-oss-120b`

## Running the Project

| Command | Description |
|---------|-------------|
| `python main.py` | CLI chat demo (asks for input, prints response + metrics) |
| `streamlit run scripts/streamlit_tokenlab.py` | Streamlit app comparing streaming vs non-streaming |
| `python scripts/practice_tokenlab.py` | Practice script (token comparison, TTFT, budget, temperature) |
| `python practice/clase3_runner.py` | Evaluate prompt versions v1–v4 with Gemma; best version on Groq |
| `python practice/clase3_runner_groq.py` | Same evaluation using Groq only, with live per-ticket output |
| `python main_rag.py` | Interactive RAG demo: load docs → chunk → index in ChromaDB → query with Groq |

## Components

### Clase 2 — Tokens, Context, Cost & Latency
- **`core/tokenlab.py`** — Token counting, latency measurement, streaming, and budget checking for Gemini and Groq.
- **Streamlit app** — Side-by-side comparison of streaming vs non-streaming responses.
- **Practice script** — Token counts (ES vs EN), TTFT experiments, `BudgetChecker` demo, and temperature experiments; writes to `outputs/`.

### Clase 3 — Prompt Engineering
- **`prompting/promptkit.py`** — Reusable toolkit: `PromptTemplate` (variable substitution + few-shot rendering), `PromptRegistry` (centralized template store), `PromptChain` (sequential prompt pipelines), and `evaluate_prompt` (accuracy, JSON parse rate, latency, token metrics against a golden set).
- **`prompting/templates/ticket_classifier.py`** — Four versions of a support-ticket classifier prompt (base → few-shot → restricted → chain) used as a running example.
- **`practice/clase3_runner.py`** — Evaluates all four prompt versions with Gemma; automatically runs the best-performing version on Groq for cross-provider comparison.
- **`practice/clase3_runner_groq.py`** — Groq-only runner that prints each ticket result in real time.
- **`practice/golden_set_tickets.json`** — Labeled dataset of support tickets used for prompt evaluation.

### Clase 4 — Structured Outputs & Function Calling
- **`structured_outputs/01–07`** — Progression from Pydantic basics through nested models, JSON mode, native structured outputs, function calling definition, the full agentic loop, and Pydantic → JSON Schema export.
- **`structured_outputs/08–11`** — Step-by-step practice building a production-grade extraction pipeline (contract → extraction → hardening → integrated).
- **`structured_outputs/actividad_extractor_recetas.py`** — Class activity: recipe extractor using structured outputs.
- **`structured_outputs/Tarea.md`** — Homework: extend models, build a `Factura` model, add retries, filter by price, and a bonus Function Calling router.

### Clase 5 — RAG Foundations
- **`rag/ingestion.py`** — Loads `.txt`, `.pdf`, and `.md` files into `Document` objects; splits them into `Chunk` objects by paragraphs (configurable `max_chunk_size`).
- **`rag/embeddings.py`** — Generates embeddings via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual, ~500 MB on first run).
- **`rag/vectorstore.py`** — Creates/opens a ChromaDB persistent collection, indexes chunks via upsert, and searches using cosine similarity.
- **`main_rag.py`** — Interactive step-by-step demo: load NovaTech corp docs → chunk → index in ChromaDB → run 5 sample queries + 1 anti-hallucination query (Groq-backed generation).
- **`data/`** — Sample corporate documents (onboarding manual, vacation policy, tech-support process, LangChain readme) used as the RAG knowledge base.

## Quick Start

```bash
cd ai-engineer-bootcamp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add GEMINI_API_KEY (or GROQ_API_KEY)
python main.py
```
