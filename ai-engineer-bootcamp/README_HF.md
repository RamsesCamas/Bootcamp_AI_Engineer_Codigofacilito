# Deploy a Hugging Face Spaces (Docker)

Free tier permanente, sin tarjeta de crédito. Ideal para portafolio.

## Costos y límites (free tier)

| Recurso | Límite |
|---|---|
| Precio | $0 |
| RAM | 16 GB |
| Disco efímero | 50 GB |
| Disco persistente | No (salvo pago) — usamos el dir `/data` del container |
| Cold start | ~30 s tras 48 h de inactividad |
| Tráfico | Sin límite razonable |

## Pasos

### 1. Crear el Space

1. Entra a <https://huggingface.co/new-space>
2. Campos:
   - **Owner**: tu usuario
   - **Space name**: `docops-agent`
   - **License**: MIT
   - **SDK**: **Docker** → *Blank*
   - **Hardware**: CPU basic · FREE
   - **Visibility**: Public (o Private si manejas docs sensibles)
3. Click **Create Space**.

### 2. Clonar el Space y copiar el código

```bash
git clone https://huggingface.co/spaces/<tu-usuario>/docops-agent
cd docops-agent

# Copia los archivos del repo del bootcamp (sin .git, sin venv, sin .env):
rsync -av --exclude='.git' --exclude='venv' --exclude='.env' \
      --exclude='chroma_db' --exclude='ops/phoenix_data' \
      /path/al/repo/ ./
```

### 3. Configurar secrets

En el Space → **Settings** → **Variables and secrets**:

| Tipo | Key | Value |
|---|---|---|
| Secret | `GROQ_API_KEY` | `gsk_…` (tu key de Groq) |
| Variable | `TRACING_ENABLED` | `false` |
| Variable | `GROQ_MODEL` | `openai/gpt-oss-120b` (o el que uses) |

### 4. Push y deploy

```bash
git add .
git commit -m "deploy: initial docops-agent"
git push
```

El Space compila el Dockerfile automáticamente. Primer build: ~5-8 min.
Cuando termine, tu app estará en `https://<tu-usuario>-docops-agent.hf.space`.

### 5. Verificación rápida

- La UI carga sin errores.
- El cuadro de chat acepta una query normal ("¿qué hace DocOps Agent?").
- El ejemplo de prompt-injection del dropdown se **bloquea** (guardrail).
- El ejemplo de PII devuelve texto con `[EMAIL]` / `[PHONE]` redactados.

## Troubleshooting

**Build falla con OOM al instalar `torch`/`sentence-transformers`:**
Usa CPU basic (no Free Zero) y añade `--no-cache-dir` (ya está en el Dockerfile).

**La app arranca pero falla al primer query con `GROQ_API_KEY missing`:**
Verifica que el secret esté en *Settings → Variables and secrets* (no solo en `.env`), y reinicia el Space (*Settings → Factory reset*).

**ChromaDB se borra al restart:**
El free tier no tiene disco persistente. Los embeddings se re-indexan desde `/data` al iniciar. Para persistencia real → upgrade a "Persistent Storage" ($0.10/GB/mes).

## Limitaciones free tier a mencionar en el README principal

- **Cold start**: ~30 s después de inactividad prolongada.
- **Sin disco persistente**: la colección de Chroma se reconstruye al bootear.
- **CPU only**: sin aceleración por GPU; el reranking con cross-encoder es el cuello de botella.
