# gemini_file_search_api.py
"""
FastAPI backend for Gemini File Search RAG (Option A: uploadToFileSearchStore).
- Directly uploads files into a File Search store (no separate Files API).
- Waits for the long-running import operation to complete.
- Reliably discovers the created document via REST `documents` list.
- Returns a stable `document_id` and `document_resource` for each file.
- Integrates cleanly with your existing service_provider.py and user.py.
"""

import os
import time
import json
import shutil
import re
import requests
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import aiofiles
from datetime import datetime

# Try to import google genai SDK; endpoints will error clearly if it's missing.
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------- CONFIG ----------------
DATA_FILE = "/data/gemini_stores.json"
UPLOAD_ROOT = Path("/data/uploads")       # temporary local storage during upload
MAX_FILE_BYTES = 50 * 1024 * 1024         # 50 MB default limit (can be overridden using "limit" flag)
POLL_INTERVAL = 2                         # seconds between polling long-running operations
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"
# ----------------------------------------

app = FastAPI(title="Gemini File Search RAG API (Option A)")

# ---------------- Helpers: persistence ----------------

def ensure_dirs():
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

def load_data() -> Dict[str, Any]:
    if not os.path.exists(DATA_FILE):
        return {"file_stores": {}, "current_store_name": None}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data: Dict[str, Any]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

ensure_dirs()
if not os.path.exists(DATA_FILE):
    save_data({"file_stores": {}, "current_store_name": None})

# ---------------- Request models ----------------

class CreateStoreRequest(BaseModel):
    api_key: str
    store_name: str   # logical name you pass from your backend (e.g. municipal_services_hubli)

class AskRequest(BaseModel):
    api_key: str
    stores: List[str]           # logical store names (same as store_name used in create_store)
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
    This follows the official pattern:
      while not op.done:
          time.sleep(...)
          op = client.operations.get(op)
    """
    op = operation
    while not getattr(op, "done", False):
        time.sleep(POLL_INTERVAL)
        try:
            if hasattr(client, "operations") and hasattr(client.operations, "get"):
                op = client.operations.get(op)
        except Exception:
            # ignore refresh errors and continue polling
            pass

    # If the operation has an error field, surface it
    if getattr(op, "error", None):
        raise RuntimeError(f"Operation failed: {op.error}")

    return op

# ---------------- REST helper to list documents for a store ----------------

def rest_list_documents_for_store(file_search_store_name: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Call Gemini REST to list documents for a File Search store.
    Returns list of dicts with at least 'name' and optionally 'displayName', 'updateTime'.
    Example:
      name: "fileSearchStores/xxx/documents/abc123"
    """
    url = f"{GEMINI_REST_BASE}/{file_search_store_name}/documents"
    params = {"key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("documents", []) or []
    except Exception:
        return []

def _parse_rfc3339_timestamp(ts: str) -> float:
    """Parse RFC3339 timestamp to epoch seconds; return 0 on failure."""
    if not ts:
        return 0.0
    try:
        # Example: "2025-01-01T12:34:56.789Z"
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0

# ---------------- Filename sanitization ----------------

def clean_filename(name: str, max_len: int = 180) -> str:
    """
    Sanitize filenames to avoid server/API header issues.
    Rules:
      - strip leading/trailing whitespace
      - normalize internal whitespace -> single underscore
      - replace disallowed chars with underscore
      - collapse consecutive underscores/dots
      - truncate to max_len
      - fallback to 'file' if empty
    """
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

    return name or "file"

# =====================================================
# CREATE STORE (creates a Gemini File Search store)
# =====================================================
@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    """
    Create a File Search store on Gemini.
    Caller passes:
      - api_key: Gemini API key
      - store_name: logical name you use in your app (e.g. municipal_services_hubli)

    Response:
      - store_name: same logical name
      - file_search_store_resource: Gemini resource (fileSearchStores/xxx)
    """
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
        return JSONResponse(
            {"error": f"Failed to create File Search store on Gemini: {e}"},
            status_code=500
        )

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
# UPLOAD files (index into the File Search store)
# =====================================================
@app.post("/stores/{store_name}/upload")
async def upload_files(
    store_name: str,
    api_key: str = Form(...),
    limit: Optional[bool] = Form(True),
    files: List[UploadFile] = File(...)
):
    """
    Upload and index files to the given store using uploadToFileSearchStore.
    Returns:
      {
        "success": bool,  # True only if all files uploaded & indexed correctly
        "results": [
          {
            "filename": "...",
            "uploaded": bool,
            "indexed": bool,
            "document_resource": "fileSearchStores/xxx/documents/yyy" | null,
            "document_id": "yyy" | null,
            "gemini_error": "..." | null,
            "reason": "... optional reason"
          },
          ...
        ]
      }
    """
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

    # temp folder for this upload - cleaned per file
    temp_folder = UPLOAD_ROOT / store_name
    temp_folder.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # Make sure "limit" is interpreted as bool even if string form is posted
    if isinstance(limit, str):
        limit_flag = limit.strip().lower() in ("true", "1", "yes", "on")
    else:
        limit_flag = bool(limit)

    for upload in files:
        original_filename = upload.filename or "file"
        filename = clean_filename(original_filename)
        temp_path = temp_folder / filename

        size = 0
        # ---- write the file locally in streaming chunks ----
        try:
            async with aiofiles.open(temp_path, "wb") as out_f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit_flag and size > MAX_FILE_BYTES:
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

            if results and results[-1].get("filename") == filename and not results[-1].get("uploaded", True):
                # This file was rejected due to size; skip further processing
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
            # Upload & index directly into File Search
            # You can add chunking_config/custom_metadata here if needed.
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=fs_store_name,
                config={
                    "display_name": filename,
                    # Example custom chunking (optional):
                    # "chunking_config": {
                    #     "white_space_config": {
                    #         "max_tokens_per_chunk": 200,
                    #         "max_overlap_tokens": 20
                    #     }
                    # }
                }
            )

            # Wait until import is complete
            op = wait_for_operation(client, op)

            # Try to fetch documents from REST and match this file
            docs = rest_list_documents_for_store(fs_store_name, api_key)
            matched_doc = None

            # 1) Best-case: match by displayName
            for d in docs:
                display = d.get("displayName") or d.get("display_name") or ""
                if display == filename:
                    matched_doc = d
                    break

            # 2) Fallback: pick the most recently updated document
            if not matched_doc and docs:
                matched_doc = max(
                    docs,
                    key=lambda d: _parse_rfc3339_timestamp(d.get("updateTime", ""))
                )

            if matched_doc:
                document_resource = matched_doc.get("name")
                if document_resource:
                    document_id = document_resource.split("/")[-1]
                    indexed_ok = True
                else:
                    gemini_error = "Document resource name missing in REST response."
                    indexed_ok = False
            else:
                gemini_error = "No documents found in File Search store after upload."
                indexed_ok = False

        except Exception as e:
            gemini_error = str(e)
            indexed_ok = False

        # Remove local temp file
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass

        # Record metadata locally (for debugging & store listing)
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

    # success=True only if ALL uploaded files were indexed correctly
    all_ok = True
    for r in results:
        if not r.get("uploaded") or not r.get("indexed"):
            all_ok = False
            break

    return {"success": all_ok, "results": results}

# =====================================================
# LIST STORES (with files metadata)
# =====================================================
@app.get("/stores")
def list_stores(api_key: str):
    """
    List local stores and indexed files.
    Validates the provided Gemini API key by trying to init the client.
    """
    try:
        init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    return {"success": True, "stores": list(data["file_stores"].values())}

# =====================================================
# DELETE DOCUMENT (REST) – used by service_provider.delete_file
# =====================================================
@app.delete("/stores/{store_name}/documents/{document_id}")
def delete_document(store_name: str, document_id: str, api_key: str):
    """
    Delete a single document from the File Search store using REST:
      DELETE https://generativelanguage.googleapis.com/v1beta/{fileSearchStore}/documents/{document_id}?force=true&key={api_key}
    Also removes this document entry from local metadata.
    """
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
        return JSONResponse(
            {"success": False, "error": f"REST request failed: {e}"},
            status_code=500
        )

    if resp.status_code not in (200, 204):
        return JSONResponse(
            {"success": False, "error": f"Gemini REST DELETE failed: {resp.text}"},
            status_code=resp.status_code
        )

    # remove local metadata entry if present
    meta["files"] = [f for f in meta.get("files", []) if f.get("document_id") != document_id]
    data["file_stores"][store_name] = meta
    save_data(data)

    return {"success": True, "deleted_document_id": document_id}

# =====================================================
# DELETE ENTIRE STORE (remote + local)
# =====================================================
@app.delete("/stores/{store_name}")
def delete_store(store_name: str, api_key: str):
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    meta = data["file_stores"][store_name]
    fs_store = meta.get("file_search_store_name")

    # Best-effort delete on Gemini
    try:
        client = init_gemini_client(api_key)
        client.file_search_stores.delete(name=fs_store, config={"force": True})
    except Exception:
        # swallow exceptions - we still remove local metadata
        pass

    # Delete local folder
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
# ASK QUESTION (RAG) - uses Gemini File Search tool
# =====================================================
@app.post("/ask")
def ask_question(payload: AskRequest):
    """
    RAG endpoint used by your main backend (user.py).

    Input:
      - api_key: Gemini API key (GKEY from env in main backend)
      - stores: list of logical store names (e.g. ["municipal_services_hubli"])
      - question: user question
      - system_prompt: optional system instruction

    Output:
      {
        "success": True/False,
        "response_text": "...",
        "grounding_metadata": {...}   # can be None
      }
    """
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    fs_store_names: List[str] = []
    for s in payload.stores:
        if s in data["file_stores"]:
            v = data["file_stores"][s].get("file_search_store_name")
            if v:
                fs_store_names.append(v)

    if not fs_store_names:
        return JSONResponse(
            {"error": "No valid File Search stores found for provided store names."},
            status_code=400
        )

    try:
        file_search_tool = types.Tool(
            file_search=types.FileSearch(file_search_store_names=fs_store_names)
        )

        system_instruction = payload.system_prompt or (
            "You are a bot assisting Indian citizens. "
            "Answer ONLY using File Search documents. "
            "If info is missing, explicitly say: "
            "'Sorry, no information available. Would you like to create a ticket?'"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=payload.question,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[file_search_tool],
                temperature=0.2
            )
        )

        grounding = None
        if hasattr(response, "candidates") and response.candidates:
            grounding = getattr(response.candidates[0], "grounding_metadata", None)

        return {
            "success": True,
            "response_text": getattr(response, "text", "") or "",
            "grounding_metadata": grounding
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
