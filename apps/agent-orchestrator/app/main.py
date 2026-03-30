import os
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel


MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:5000")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")


class ChatRequest(BaseModel):
    message: str
    image_url: str | None = None
    session_id: str | None = None


app = FastAPI(title="Agent Orchestrator", version="0.1.0")


# Lightweight in-memory session store for demo flow.
SESSIONS: dict[str, dict[str, Any]] = {}
MAX_HISTORY = 12


def confidence_band(conf: float) -> str:
    if conf >= 0.85:
        return "high"
    if conf >= 0.60:
        return "medium"
    return "low"


def ambiguity_margin(top_predictions: list[dict[str, Any]]) -> float:
    if len(top_predictions) < 2:
        return 1.0
    return float(top_predictions[0]["confidence"]) - float(top_predictions[1]["confidence"])


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    with httpx.Client(timeout=90.0) as client:
        resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data.get("message", {}).get("content", "")


def compose_answer(
    message: str,
    retrieval_items: list[dict[str, Any]],
    diagnosis: dict[str, Any] | None,
    policy_note: str,
    session_context: dict[str, Any] | None = None,
) -> str:
    context_block = "\n\n".join(
        [f"- Source: {item.get('source')} | Score: {item.get('score')}\n{item.get('text')}" for item in retrieval_items]
    )
    diagnosis_block = "No image-based diagnosis available."
    if diagnosis:
        diagnosis_block = (
            f"Predicted class: {diagnosis.get('class')}\n"
            f"Plant: {diagnosis.get('plant')}\n"
            f"Disease: {diagnosis.get('disease')}\n"
            f"Confidence: {diagnosis.get('confidence')}"
        )

    system_prompt = (
        "You are an agronomy assistant. Use evidence first, be concise, and include next steps. "
        "If confidence is low or ambiguous, avoid definitive claims."
    )
    user_prompt = (
        f"User request:\n{message}\n\n"
        f"Diagnosis context:\n{diagnosis_block}\n\n"
        f"Session context:\n{session_context or {}}\n\n"
        f"Policy:\n{policy_note}\n\n"
        f"Retrieved evidence:\n{context_block}\n\n"
        "Produce a practical answer with: diagnosis status, explanation, immediate actions, and monitoring advice."
    )
    try:
        text = call_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
        if text.strip():
            return text.strip()
    except Exception:
        pass

    if not retrieval_items:
        fallback = "I could not generate a model-based final narrative."
        if diagnosis:
            fallback += (
                f" Predicted {diagnosis.get('disease')} for {diagnosis.get('plant')} "
                f"with confidence {diagnosis.get('confidence'):.3f}."
            )
        return fallback

    primary = retrieval_items[0]
    source = str(primary.get("source", "knowledge base"))
    text = str(primary.get("text", "")).replace("\n", " ").strip()
    text = " ".join(text.split())
    excerpt = text[:520].rstrip()
    if len(text) > 520:
        excerpt += "..."

    answer_lines: list[str] = []
    if diagnosis:
        disease = diagnosis.get("disease", "unknown disease")
        plant = diagnosis.get("plant", "plant")
        confidence = float(diagnosis.get("confidence", 0.0))
        answer_lines.append(
            f"Image-based prediction: {disease} on {plant} (confidence {confidence:.3f})."
        )
        answer_lines.append("Guidance based on retrieved knowledge:")
    else:
        answer_lines.append(f"Based on the available knowledge base, here is relevant guidance for: {message}")

    answer_lines.append(excerpt)

    if len(retrieval_items) > 1:
        answer_lines.append("Additional relevant sources were considered to reduce uncertainty.")

    return "\n".join(answer_lines)


def run_diagnosis_flow(
    message: str,
    image_bytes: bytes,
    filename: str = "leaf.jpg",
    session_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=60.0) as client:
            files = {"image": (filename, image_bytes, "image/jpeg")}
            pred_resp = client.post(f"{MODEL_SERVICE_URL}/predict", files=files)
            pred_resp.raise_for_status()
            pred_data = pred_resp.json()
    except httpx.HTTPStatusError as exc:
        detail: str | dict[str, Any] | None = None
        try:
            payload = exc.response.json()
            detail = payload.get("error") if isinstance(payload, dict) else payload
        except Exception:
            detail = exc.response.text.strip()[:500]

        return {
            "mode": "image_prediction_error",
            "final_answer": (
                "Image diagnosis service returned an error. "
                "Check model-service health and ensure model artifacts are available."
            ),
            "upstream_service": "model-service",
            "upstream_status": exc.response.status_code,
            "upstream_error": detail,
        }
    except httpx.RequestError as exc:
        return {
            "mode": "image_prediction_error",
            "final_answer": (
                "Image diagnosis service is unreachable. "
                "Check that model-service is running and accessible from agent-orchestrator."
            ),
            "upstream_service": "model-service",
            "upstream_error": str(exc),
        }

    if not pred_data.get("success"):
        return {
            "mode": "image_prediction_error",
            "final_answer": "Prediction service returned an error. Please retry with a clear image.",
            "prediction": pred_data,
        }

    prediction = pred_data.get("prediction", {})
    top5 = pred_data.get("top_5_predictions", [])
    top_conf = float(prediction.get("confidence", 0.0))
    band = confidence_band(top_conf)
    margin = ambiguity_margin(top5)
    ambiguous = margin < 0.15

    rag_top_k = 3
    if band == "medium" or ambiguous:
        rag_top_k = 5

    with httpx.Client(timeout=20.0) as client:
        rag_resp = client.post(
            f"{RAG_SERVICE_URL}/retrieve",
            json={
                "query": message,
                "disease_label": prediction.get("class"),
                "top_k": rag_top_k,
            },
        )
        rag_resp.raise_for_status()
        rag_data = rag_resp.json()

    if band == "high" and not ambiguous:
        policy_note = "High confidence and clear margin. You may provide focused guidance grounded in evidence."
        diagnosis_status = "confirmed"
    elif band == "low":
        policy_note = "Low confidence. Do not provide definitive diagnosis. Provide hypotheses and request better image quality."
        diagnosis_status = "tentative"
    else:
        policy_note = "Medium confidence or ambiguous result. Provide cautious differential guidance and ask for additional evidence."
        diagnosis_status = "tentative"

    final_answer = compose_answer(
        message=message,
        retrieval_items=rag_data.get("items", []),
        diagnosis=prediction,
        policy_note=policy_note,
        session_context=session_ctx,
    )

    return {
        "mode": "diagnose_and_explain",
        "diagnosis_status": diagnosis_status,
        "confidence_band": band,
        "ambiguity_margin": margin,
        "ambiguous": ambiguous,
        "prediction": prediction,
        "top_5_predictions": top5,
        "retrieval": rag_data,
        "final_answer": final_answer,
        "next_action": "If uncertain, upload a sharper image with good lighting from both sides of one leaf.",
    }


def get_or_create_session(session_id: str | None) -> tuple[str, dict[str, Any]]:
    sid = session_id or str(uuid4())
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "history": [],
            "last_prediction": None,
            "last_retrieval": None,
        }
    return sid, SESSIONS[sid]


def append_history(session: dict[str, Any], role: str, text: str) -> None:
    session["history"].append({"role": role, "text": text})
    if len(session["history"]) > MAX_HISTORY:
        session["history"] = session["history"][-MAX_HISTORY:]


def session_context(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "last_prediction": session.get("last_prediction"),
        "history_tail": session.get("history", [])[-4:],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "service": "agent-orchestrator",
        "chat_model": OLLAMA_CHAT_MODEL,
    }


def handle_chat(
    message: str,
    session_id: str | None,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
    filename: str = "leaf.jpg",
) -> dict[str, Any]:
    sid, session = get_or_create_session(session_id)
    append_history(session, "user", message)

    # Text-only path: retrieve knowledge and answer with local LLM.
    if not image_url and not image_bytes:
        disease_hint = None
        if session.get("last_prediction"):
            disease_hint = session["last_prediction"].get("class")

        with httpx.Client(timeout=20.0) as client:
            rag_resp = client.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={"query": message, "disease_label": disease_hint, "top_k": 3},
            )
            rag_resp.raise_for_status()
            rag_data = rag_resp.json()

        session["last_retrieval"] = rag_data

        final_answer = compose_answer(
            message=message,
            retrieval_items=rag_data.get("items", []),
            diagnosis=None,
            policy_note="No image provided. Give informational guidance and request an image for diagnosis.",
            session_context=session_context(session),
        )
        append_history(session, "assistant", final_answer)

        return {
            "session_id": sid,
            "mode": "rag_only",
            "final_answer": final_answer,
            "retrieval": rag_data,
            "next_action": "Upload a clear leaf image for disease diagnosis."
        }

    resolved_image_bytes = image_bytes

    if resolved_image_bytes is None and image_url:
        # Image URL path: fetch image and forward to model-service /predict.
        with httpx.Client(timeout=60.0) as client:
            img_resp = client.get(image_url)
            img_resp.raise_for_status()
        resolved_image_bytes = img_resp.content

    if not resolved_image_bytes:
        return {
            "session_id": sid,
            "mode": "image_prediction_error",
            "final_answer": "Empty image file. Please upload a valid image.",
        }

    result = run_diagnosis_flow(
        message=message,
        image_bytes=resolved_image_bytes,
        filename=filename,
        session_ctx=session_context(session),
    )
    if result.get("mode") == "diagnose_and_explain":
        session["last_prediction"] = result.get("prediction")
        session["last_retrieval"] = result.get("retrieval")
        append_history(session, "assistant", str(result.get("final_answer", "")))

    result["session_id"] = sid
    return result


def compact_chat_response(result: dict[str, Any]) -> dict[str, Any]:
    response: dict[str, Any] = {
        "session_id": result.get("session_id"),
        "mode": result.get("mode"),
        "final_answer": result.get("final_answer"),
    }

    next_action = result.get("next_action")
    if next_action:
        response["next_action"] = next_action

    if result.get("diagnosis_status"):
        response["diagnosis_status"] = result.get("diagnosis_status")
    if result.get("confidence_band"):
        response["confidence_band"] = result.get("confidence_band")

    prediction = result.get("prediction")
    if isinstance(prediction, dict):
        response["prediction"] = {
            "plant": prediction.get("plant"),
            "disease": prediction.get("disease"),
            "confidence": prediction.get("confidence"),
        }

    if result.get("mode") == "image_prediction_error":
        if result.get("upstream_status") is not None:
            response["upstream_status"] = result.get("upstream_status")
        if result.get("upstream_error") is not None:
            response["upstream_error"] = result.get("upstream_error")

    return response


async def parse_and_handle_chat(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        payload = ChatRequest.model_validate(await request.json())
        return handle_chat(
            message=payload.message,
            session_id=payload.session_id,
            image_url=payload.image_url,
        )

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        message = str(form.get("message", "")).strip()
        session_id_raw = form.get("session_id")
        session_id = str(session_id_raw).strip() if session_id_raw else None
        image_url_raw = form.get("image_url")
        image_url = str(image_url_raw).strip() if image_url_raw else None

        if not message:
            raise HTTPException(status_code=422, detail="Field 'message' is required")

        image = form.get("image")
        image_bytes: bytes | None = None
        filename = "leaf.jpg"
        if isinstance(image, UploadFile):
            image_bytes = await image.read()
            filename = image.filename or filename

        return handle_chat(
            message=message,
            session_id=session_id,
            image_url=image_url,
            image_bytes=image_bytes,
            filename=filename,
        )

    raise HTTPException(
        status_code=415,
        detail="Unsupported Content-Type. Use application/json or multipart/form-data.",
    )


@app.post("/chat")
async def chat(request: Request) -> dict[str, Any]:
    return await parse_and_handle_chat(request)


@app.post("/chat-answer")
async def chat_answer(request: Request) -> dict[str, Any]:
    result = await parse_and_handle_chat(request)
    return compact_chat_response(result)


@app.post("/chat-upload")
async def chat_upload(
    message: str = Form(...),
    image: UploadFile = File(...),
    session_id: str | None = Form(default=None),
) -> dict[str, Any]:
    image_bytes = await image.read()
    filename = image.filename or "leaf.jpg"
    return handle_chat(
        message=message,
        session_id=session_id,
        image_bytes=image_bytes,
        filename=filename,
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in SESSIONS:
        return {"success": False, "error": "session_not_found"}
    return {"success": True, "session_id": session_id, "state": SESSIONS[session_id]}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    existed = session_id in SESSIONS
    if existed:
        del SESSIONS[session_id]
    return {"success": True, "session_id": session_id, "deleted": existed}
