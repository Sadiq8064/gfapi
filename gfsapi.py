# gemini_file_search_api.py
"""
FastAPI backend for REAL Gemini File Search RAG (Option A: delete local temp file after indexing).
- Keeps document_id logic (SDK + REST fallback).
- Sanitizes filenames before uploading to avoid "Illegal header value" errors.
- Deletes local temp file after successful (or attempted) indexing.
- NOW SUPPORTS STREAMING on the existing /ask endpoint (POST /ask unchanged)
"""
import os
import time
import json
import shutil
import re
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import aiofiles
import asyncio

# Try to import google genai SDK; endpoints will error clearly if it's missing.
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------- CONFIG ----------------
DATA_FILE = "/data/gemini_stores.json"
UPLOAD_ROOT = Path("/data/uploads")
        # temporary local storage during upload
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB default limit (can be skipped via form)
POLL_INTERVAL = 2  # seconds between polling long-running operations
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"
# ----------------------------------------

app = FastAPI(title="Gemini File Search RAG API (Option A)")

# ---------------- Helpers: persistence ----------------
def ensure_dirs():
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"file_stores": {}, "current_store_name": None}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

ensure_dirs()
if not os.path.exists(DATA_FILE):
    save_data({"file_stores": {}, "current_store_name": None})

# ---------------- Request models ----------------
class CreateStoreRequest(BaseModel):
    api_key: str
    store_name: str

class AskRequest(BaseModel):
    api_key: str
    stores: List[str]
    question: str
    system_prompt: Optional[str] = None

# ---------------- Gemini client helper ----------------
def init_gemini_client(api_key: str):
    """Initialize google genai client (per-request). Raises on missing SDK or invalid key."""
    if genai is None:
        raise RuntimeError("google-genai SDK is not installed on the server.")
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

def wait_for_operation(client, operation):
    """
    Poll a long-running operation object until done.
    Works with SDK operation objects that support `.done` and client.operations.get(op).
    """
    op = operation
    while not getattr(op, "done", False):
        time.sleep(POLL_INTERVAL)
        try:
            if hasattr(client, "operations") and hasattr(client.operations, "get"):
                op = client.operations.get(op)
        except Exception:
            pass
    if getattr(op, "error", None):
        raise RuntimeError(f"Operation failed: {op.error}")
    return op

# ---------------- REST helper to list documents for a store ----------------
def rest_list_documents_for_store(file_search_store_name: str, api_key: str):
    url = f"{GEMINI_REST_BASE}/{file_search_store_name}/documents"
    params = {"key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("documents", [])
    except Exception:
        return []

# ---------------- Filename sanitization ----------------
def clean_filename(name: str, max_len: int = 180) -> str:
    if not name:
        return "file"
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"^\.+", "", name)
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "_", name)
    name = re.sub(r"__+", "_", name)
    name = re.sub(r"\.\.+", ".", name)
    if len(name) > max_len:
        name = name[:max_len]
    if not name:
        return "file"
    return name

# ---------------- Utility ----------------
def _build_example_delete_url(store_name: str, document_id: Optional[str]):
    return f"DELETE /stores/{store_name}/documents/{document_id}?api_key=YOUR_API_KEY"

# =====================================================
# CREATE STORE
# =====================================================
@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    if payload.store_name in data["file_stores"]:
        return JSONResponse({"error": "A store with this name already exists."}, status_code=400)

    try:
        fs_store = client.file_search_stores.create(config={"display_name": payload.store_name})
        fs_store_name = getattr(fs_store, "name", None) or fs_store
    except Exception as e:
        return JSONResponse({"error": f"Failed to create File Search store on Gemini: {e}"}, status_code=500)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    data["file_stores"][payload.store_name] = {
        "store_name": payload.store_name,
        "file_search_store_name": fs_store_name,
        "created_at": created_at,
        "files": []
    }
    data["current_store_name"] = payload.store_name
    save_data(data)

    return {
        "success": True,
        "store_name": payload.store_name,
        "file_search_store_resource": fs_store_name,
        "created_at": created_at,
        "file_count": 0
    }

# =====================================================
# UPLOAD files
# =====================================================
@app.post("/stores/{store_name}/upload")
async def upload_files(
    store_name: str,
    api_key: str = Form(...),
    limit: Optional[bool] = Form(True),
    files: List[UploadFile] = File(...)
):
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    store_meta = data["file_stores"][store_name]
    fs_store_name = store_meta.get("file_search_store_name")
    if not fs_store_name:
        raise HTTPException(status_code=500, detail="File Search store mapping missing")

    temp_folder = UPLOAD_ROOT / store_name
    temp_folder.mkdir(parents=True, exist_ok=True)
    results = []

    for upload in files:
        original_filename = upload.filename or "file"
        filename = clean_filename(original_filename)
        temp_path = temp_folder / filename

        size = 0
        try:
            async with aiofiles.open(temp_path, "wb") as out_f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > MAX_FILE_BYTES:
                        await out_f.close()
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                        results.append({
                            "filename": filename,
                            "uploaded": False,
                            "indexed": False,
                            "reason": f"File exceeds limit of {MAX_FILE_BYTES} bytes (50MB)."
                        })
                        break
                    await out_f.write(chunk)
            if results and results[-1].get("filename") == filename and results[-1].get("uploaded") is False:
                continue
        except Exception as e:
            results.append({
                "filename": filename,
                "uploaded": False,
                "indexed": False,
                "reason": f"Failed to save local file: {e}"
            })
            continue

        document_resource = None
        document_id = None
        indexed_ok = False
        gemini_error = None
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=fs_store_name,
                config={"display_name": filename}
            )
            op = wait_for_operation(client, op)
            try:
                document_resource = op.response.file_search_document.name
            except Exception:
                document_resource = None

            if not document_resource:
                docs = rest_list_documents_for_store(fs_store_name, api_key)
                for d in docs:
                    display = d.get("displayName") or d.get("display_name") or ""
                    if display == filename:
                        document_resource = d.get("name")
                        break
                if not document_resource:
                    for d in docs:
                        if filename in (d.get("name") or ""):
                            document_resource = d.get("name")
                            break
            if document_resource:
                document_id = document_resource.split("/")[-1]
                indexed_ok = True
            else:
                indexed_ok = True
        except Exception as e:
            gemini_error = str(e)
            indexed_ok = False

        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass

        entry = {
            "display_name": filename,
            "size_bytes": size,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gemini_indexed": indexed_ok,
            "document_resource": document_resource,
            "document_id": document_id,
            "gemini_error": gemini_error
        }
        store_meta.setdefault("files", []).append(entry)
        save_data(data)

        results.append({
            "filename": filename,
            "uploaded": True,
            "indexed": indexed_ok,
            "document_resource": document_resource,
            "document_id": document_id,
            "gemini_error": gemini_error
        })

    return {"success": True, "results": results}

# =====================================================
# LIST STORES
# =====================================================
@app.get("/stores")
def list_stores(api_key: str):
    try:
        init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    return {"success": True, "stores": list(data["file_stores"].values())}

# =====================================================
# DELETE DOCUMENT
# =====================================================
@app.delete("/stores/{store_name}/documents/{document_id}")
def delete_document(store_name: str, document_id: str, api_key: str):
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    meta = data["file_stores"][store_name]
    fs_store = meta.get("file_search_store_name")
    if not fs_store:
        raise HTTPException(status_code=500, detail="File search store mapping missing")

    doc_resource = f"{fs_store}/documents/{document_id}"
    url = f"{GEMINI_REST_BASE}/{doc_resource}"
    params = {"force": "true", "key": api_key}

    try:
        resp = requests.delete(url, params=params, timeout=15)
    except Exception as e:
        return JSONResponse({"success": False, "error": f"REST request failed: {e}"}, status_code=500)

    if resp.status_code not in (200, 204):
        return JSONResponse({"success": False, "error": f"Gemini REST DELETE failed: {resp.text}"}, status_code=resp.status_code)

    meta["files"] = [f for f in meta.get("files", []) if f.get("document_id") != document_id]
    data["file_stores"][store_name] = meta
    save_data(data)

    return {"success": True, "deleted_document_id": document_id}

# =====================================================
# DELETE ENTIRE STORE
# =====================================================
@app.delete("/stores/{store_name}")
def delete_store(store_name: str, api_key: str):
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    meta = data["file_stores"][store_name]
    fs_store = meta.get("file_search_store_name")

    try:
        client = init_gemini_client(api_key)
        client.file_search_stores.delete(name=fs_store, config={"force": True})
    except Exception:
        pass

    folder = UPLOAD_ROOT / store_name
    if folder.exists():
        try:
            shutil.rmtree(folder)
        except Exception:
            pass

    del data["file_stores"][store_name]
    if data.get("current_store_name") == store_name:
        data["current_store_name"] = None

    save_data(data)
    return {"success": True, "deleted_store": store_name}

# =====================================================
# ASK QUESTION (RAG) - NOW STREAMING, ENDPOINT UNCHANGED: POST /ask
# =====================================================
@app.post("/ask")
async def ask_question(payload: AskRequest):
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    fs_store_names = []
    for s in payload.stores:
        if s in data["file_stores"]:
            v = data["file_stores"][s].get("file_search_store_name")
            if v:
                fs_store_names.append(v)

    if not fs_store_names:
        return JSONResponse({"error": "No valid File Search stores found for provided store names."}, status_code=400)

    try:
        file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=fs_store_names))
        system_instruction = payload.system_prompt or (
            "You are a document summarization assistant. ONLY summarize information directly found in provided File Search stores."
        )

        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=payload.question,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[file_search_tool],
                temperature=0.2
            )
        )

        async def event_stream():
            full_text = ""
            grounding = None

            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    full_text += chunk.text
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                    await asyncio.sleep(0)

                if hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, "grounding_metadata"):
                        grounding = candidate.grounding_metadata

            yield f"data: {json.dumps({'done': True, 'response_text': full_text, 'grounding_metadata': grounding})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
