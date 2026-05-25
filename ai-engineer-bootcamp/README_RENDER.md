# Deploy a Render

Buen tercer destino si ya tienes otros servicios en Render. Cold starts aceptables, pero el free tier hiberna agresivamente.

## Costos y límites (free tier)

| Recurso | Límite |
|---|---|
| Precio | $0 |
| RAM | 512 MB |
| CPU | 0.1 vCPU |
| Hibernación | Tras 15 min sin tráfico |
| Cold start | ~30-60 s |
| Disco persistente | **No** en free (solo plan Starter, $7/mes) |
| Tráfico | 100 GB/mes |

> **Advertencia crítica:** 512 MB de RAM son **marginales** para este stack (LangGraph + ChromaDB + sentence-transformers). Si OOMs al arrancar, usa HF Spaces.

## Trade-off sobre persistencia

El free tier de Render **no incluye disco persistente**. Opciones:

### Opción 1 — ChromaDB efímero (recomendado para demo)
Acepta que el índice se reconstruya en cada cold start (~20-40 s extra).
**No cambies nada** — es el comportamiento actual con `render.yaml` que te damos.

### Opción 2 — Upgrade a Starter ($7/mes)
Descomenta el bloque `disk:` en `render.yaml` y cambia `plan: free` → `plan: starter`.
El volumen de 1 GB en `/opt/render/project/src/data` persiste entre deploys y restarts.

## Pasos

### 1. Conecta tu repo

1. <https://dashboard.render.com/>
2. **New → Blueprint** → conecta GitHub → selecciona el repo.
3. Render detecta automáticamente `render.yaml`.

### 2. Configura el secret

En la pantalla de confirmación del Blueprint:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_...` (manual, no syncable) |

Las demás vars ya vienen en `render.yaml`.

### 3. Deploy

Click **Apply**. Primer build: 5-10 min (Docker build + pip install).

### 4. Verificación

- Abre la URL que te da Render (`https://docops-agent.onrender.com`).
- Espera el cold start (~30 s).
- Envía una query → debe responder.
- Revisa logs en tiempo real en el dashboard.

## Troubleshooting

**`OOMKilled` al arrancar:**
El free tier (512 MB) no alcanza. Upgrade a Starter o cambia a HF Spaces.

**Cold start de >60 s:**
Normal en free tier. Considera un **UptimeRobot** gratis (ping cada 5 min) para mantener el servicio despierto durante la clase.

**ChromaDB vacío después de restart:**
Esperado en free tier sin disco. Ver "Opción 2" arriba si necesitas persistencia.

## Limitaciones a mencionar en el README principal

- **512 MB RAM** hace que este plan sea marginal para el stack completo.
- **Sin persistencia** en free; cold start re-indexa.
- **Hibernación 15 min** genera picos de latencia en el primer request tras un lapso de inactividad.
