# Grape Agentic AI

Agentic AI system for grape disease diagnosis and explanation.

This repository is intentionally separated from the existing `plant-disease-detection` project to avoid mixing ML experimentation with orchestration and RAG concerns.

## High-Level Architecture

- `apps/model-service`: bridge to your existing model API project
- `apps/rag-service`: retrieval API over disease knowledge base using Qdrant + Ollama embeddings
- `apps/agent-orchestrator`: user-facing API with reasoning and policy checks using Ollama chat model
- `data/rag_kb`: source files for retrieval knowledge base
- `data/evaluation`: acceptance scenarios and evaluation datasets


### Components

1. `model-service`
- Existing Flask API from `plant-disease-detection`
- Returns class prediction and confidence

2. `rag-service`
- Retrieves disease context for explanation and guidance
- Uses Ollama embeddings (`nomic-embed-text`) + Qdrant vector search

3. `agent-orchestrator`
- Receives user requests
- Calls tools based on request type and confidence policy
- Applies confidence/ambiguity policy and generates final response with Ollama chat model (`qwen2.5:7b-instruct`)


## Expected Local Setup

1. Keep your current project in sibling folder:
   - `../plant-disease-detection`
2. Run this repository from:
   - `./grape-agentic-ai`

## Quick Start

1. Fill `.env` from `.env.example`
2. Start services:

```bash
docker compose up --build
```

For Podman:

```bash
cd grape-agentic-ai
podman compose up --build
```

Before testing image endpoints, ensure model files exist in sibling project:

- `../plant-disease-detection/models/plant_disease_model.keras`
- `../plant-disease-detection/models/class_indices.json`

If they are missing, image diagnosis via `/chat` will fail because `model-service /predict` returns `500`.

3. Pull Ollama models (first run only):

```bash
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull qwen2.5:7b-instruct
```

Podman equivalent:

```bash
podman exec -it ollama ollama pull nomic-embed-text
podman exec -it ollama ollama pull qwen2.5:7b-instruct
```

4. Check health endpoints:

- Orchestrator: `http://localhost:8000/health`
- RAG: `http://localhost:8001/health`
- Model service: `http://localhost:5000/health`
- Qdrant: `http://localhost:6333`
- Ollama: `http://localhost:11434`

5. Test text-only orchestration:

```bash
curl -X POST "http://localhost:8000/chat" \
   -H "Content-Type: application/json" \
   -d "{\"message\":\"What are common signs of grape black rot and first actions?\"}"
```

Save `session_id` from the response and reuse it in next requests.

If you need a compact client-friendly response without raw `retrieval.items`, use `/chat-answer`:

```bash
curl -X POST "http://localhost:8000/chat-answer" \
   -H "Content-Type: application/json" \
   -d "{\"message\":\"What are common signs of grape black rot and first actions?\"}"
```

6. Test image flow orchestration (public image URL):

```bash
curl -X POST "http://localhost:8000/chat" \
   -H "Content-Type: application/json" \
   -d "{\"message\":\"What should I do now?\", \"image_url\":\"https://example.com/leaf.jpg\", \"session_id\":\"<SESSION_ID_FROM_STEP_5>\"}"
```

7. Test direct image upload in the same `/chat` endpoint (recommended for local testing):

```bash
curl -X POST "http://localhost:8000/chat" \
   -F "message=What should I do now?" \
   -F "session_id=<SESSION_ID_FROM_STEP_5>" \
   -F "image=@/path/to/leaf.jpg"
```

Backward-compatible alias (optional): `/chat-upload` still works.

8. Inspect or reset session state:

```bash
curl "http://localhost:8000/sessions/<SESSION_ID>"
curl -X DELETE "http://localhost:8000/sessions/<SESSION_ID>"
```

## Selected Orchestrator Model

- Chat model: `qwen2.5:7b-instruct`
- Why: good balance between reasoning quality and local inference speed for orchestration tasks.
- Embeddings model: `nomic-embed-text`

## Using `data/evaluation`

Use `data/evaluation` to store acceptance scenarios for end-to-end checks of the full system, not only the classifier.

Recommended contents:

- input scenarios for `/chat` requests
- test images or references to them
- expected behavior rules
- manual or automated evaluation notes

Suggested structure:

```text
data/evaluation/
   scenarios/
      high_confidence.json
      medium_confidence.json
      low_confidence.json
      text_only.json
      service_down.json
   images/
      black_rot_01.jpg
      unclear_leaf_01.jpg
   notes/
      expected-behavior.md
```

Example scenario file:

```json
{
   "id": "low-confidence-unclear-image",
   "input": {
      "message": "What is this disease and what should I do?",
      "image_path": "data/evaluation/images/unclear_leaf_01.jpg"
   },
   "expected": {
      "diagnosis_status": "tentative",
      "must_include": [
         "uncertain",
         "next step"
      ],
      "must_not_include": [
         "definitely",
         "confirmed diagnosis"
      ]
   }
}
```

Typical scenarios to add first:

1. High-confidence image diagnosis
2. Medium-confidence ambiguous diagnosis
3. Low-confidence uncertain case with safe fallback
4. Text-only query answered with RAG only
5. Service-down case with graceful degradation

Manual evaluation flow:

1. Pick a scenario file from `data/evaluation/scenarios`
2. Send its input to `/chat`
3. Check that the response follows the rules in `expected`
4. Record pass or fail in `data/evaluation/notes`

Example manual image evaluation request:

```bash
curl -X POST "http://localhost:8000/chat" \
    -F "message=What should I do now?" \
    -F "image=@data/evaluation/images/black_rot_01.jpg"
```

Example manual text-only evaluation request:

```bash
curl -X POST "http://localhost:8000/chat-answer" \
    -H "Content-Type: application/json" \
    -d "{\"message\":\"What are common signs of grape black rot and first actions?\"}"
```

Minimum response checks for acceptance:

- diagnosis status is explicit: confirmed or tentative
- predicted disease and confidence band are present when image diagnosis is used
- explanation is grounded and short
- every answer includes a next action
- low-confidence cases do not produce a definitive diagnosis

## Next Milestones

1. Add domain KB files in `data/rag_kb`
2. Add acceptance scenarios in `data/evaluation`
3. Add integration tests for end-to-end flow
4. Add reflection scoring and retry strategy in orchestrator
