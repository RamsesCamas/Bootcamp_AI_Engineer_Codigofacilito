.PHONY: test-contracts test-goldens test-guardrails test eval demo-clase14 \
        demo-clase16 clean-results phoenix-up phoenix-down release-check run-ui

test-contracts:
	python -m pytest tests/test_contracts.py -v

test-goldens:
	python -m pytest tests/test_goldens.py -v --tb=short

test-guardrails:
	python -m pytest tests/test_guardrails.py -v

test:
	python -m pytest tests/ -v

eval:
	python -m evals.runner

demo-clase14:
	python -m demos.clase14_demo

demo-clase16:
	python scripts/demo_class_16.py

clean-results:
	rm -rf evals/results/*.json

# ── Clase 16: Observabilidad ──────────────────────────────
phoenix-up:
	docker compose -f ops/docker-compose.phoenix.yml up -d
	@echo "Phoenix UI: http://localhost:6006"

phoenix-down:
	docker compose -f ops/docker-compose.phoenix.yml down

# ── Clase 16: UI ──────────────────────────────────────────
run-ui:
	streamlit run streamlit_app.py

# ── Clase 16: Pre-deploy ──────────────────────────────────
release-check:
	python scripts/release_check.py
