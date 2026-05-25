# Deploy a Streamlit Community Cloud

Free tier permanente, conecta a GitHub. El más rápido de los tres.

## Costos y límites (free tier)

| Recurso | Límite |
|---|---|
| Precio | $0 |
| RAM | 1 GB |
| CPU | Compartida |
| Cold start | ~10-20 s tras inactividad |
| Repo | Requiere GitHub público (o privado con enlace de OAuth) |

> **Restricción importante:** 1 GB es poco para `sentence-transformers` + ChromaDB + LangGraph. Si el cold-start falla por OOM, usa **HF Spaces** (16 GB) o **Render** en su lugar.

## Pasos

### 1. Prepara el repo en GitHub

1. Asegúrate de que `streamlit_app.py` está en la raíz.
2. Confirma que `requirements.txt` incluye las deps de la clase 16 (streamlit, arize-phoenix, openinference-…, etc.).
3. `.streamlit/config.toml` ya está en el repo — define tema oscuro + headless.
4. Push a la rama que quieras deployar.

### 2. Deploy

1. Entra a <https://share.streamlit.io>.
2. **New app** → elige repo + rama + archivo `streamlit_app.py`.
3. **Advanced settings**:
   - **Python version**: `3.11`
   - **Secrets**: pega el contenido de `.streamlit/secrets.toml.example` con tus valores reales:
     ```toml
     GROQ_API_KEY = "gsk_..."
     GROQ_MODEL = "openai/gpt-oss-120b"
     TRACING_ENABLED = "false"
     CHROMA_PATH = "/mount/src/repo/data/chroma"
     ```
4. **Deploy**. Primer build: ~3-5 min.

### 3. Acceso a los secrets desde el código

Streamlit Cloud expone los secrets como `st.secrets` y también como env vars.
Nuestro código usa `os.getenv(...)` → `st.secrets` se auto-propaga, **no hace falta** cambiar nada.

### 4. Verificación

- Abre la URL pública (`https://<tu-app>.streamlit.app`).
- Envía una query normal y verifica respuesta.
- Prueba el dropdown "Injection: ignore previous" → debe bloquearse.

## Troubleshooting

**Build falla con `Killed` al instalar dependencias:**
Out-of-memory en el build. Opciones:
1. Elimina `rank_bm25` y `sentence-transformers` (degrada el RAG al vector search puro).
2. Migra a HF Spaces (16 GB).

**ChromaDB se borra entre deploys:**
Streamlit Cloud no tiene storage persistente. Los embeddings se recrean al bootear. Si tu corpus es grande considera precomputar el índice y versionarlo con Git LFS.

**Cold start lento:**
Es esperado — 10-20 s en el primer request tras inactividad. La app permanece "warm" mientras haya tráfico.

## Limitaciones a mencionar en el README principal

- **1 GB RAM**: no caben modelos `sentence-transformers` grandes.
- **Sin persistencia**: cada deploy reindexea el corpus.
- **CPU compartida**: latencias menos predecibles que HF Spaces.
