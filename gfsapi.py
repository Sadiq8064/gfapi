# gemini_file_search_api.py
"""
FastAPI backend for REAL Gemini File Search RAG (Option A: delete local temp file after indexing).
- Keeps document_id logic (SDK + REST fallback).
- Sanitizes filenames before uploading to avoid "Illegal header value" errors.
- Deletes local temp file after successful (or attempted) indexing.
"""

import os
import time
import json
import shutil
import re
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import aiofiles

# Try to import google genai SDK; endpoints will error clearly if it's missing.
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------- CONFIG ----------------
DATA_FILE = "gemini_stores.json"
UPLOAD_ROOT = Path("uploads")           # temporary local storage during upload
MAX_FILE_BYTES = 50 * 1024 * 1024      # 50 MB default limit (can be skipped via form)
POLL_INTERVAL = 2                       # seconds between polling long-running operations
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
            # refresh operation if possible
            if hasattr(client, "operations") and hasattr(client.operations, "get"):
                op = client.operations.get(op)
        except Exception:
            # ignore refresh errors and continue polling
            pass

    if getattr(op, "error", None):
        raise RuntimeError(f"Operation failed: {op.error}")

    return op

# ---------------- REST helper to list documents for a store ----------------

def rest_list_documents_for_store(file_search_store_name: str, api_key: str):
    """
    Call Gemini REST to list documents for a File Search store.
    Returns list of dicts with at least 'name' and optionally 'displayName'.
    Example response.documents[i].name -> "fileSearchStores/xxx/documents/abc123"
    """
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
    """
    Sanitize filenames to avoid server/API header issues.
    Rules applied:
      - strip leading/trailing whitespace
      - normalize internal whitespace -> single underscore
      - replace disallowed characters with underscore
      - collapse consecutive underscores
      - truncate to max_len
      - avoid empty result (fallback to 'file')
    """
    if not name:
        return "file"

    # Normalize unicode and strip
    name = str(name).strip()

    # Replace any whitespace sequence with single underscore
    name = re.sub(r"\s+", "_", name)

    # Remove leading dots (avoid hidden files like .env)
    name = re.sub(r"^\.+", "", name)

    # Keep only safe characters: letters, digits, underscores, hyphens, dots
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "_", name)

    # Collapse multiple underscores or dots
    name = re.sub(r"__+", "_", name)
    name = re.sub(r"\.\.+", ".", name)

    # Trim to max length
    if len(name) > max_len:
        name = name[:max_len]

    # Final fallback
    if not name:
        return "file"
    return name

# ---------------- Utility: build delete instruction template (NOT returned on upload per your request) ----------------

def _build_example_delete_url(store_name: str, document_id: Optional[str]):
    return f"DELETE /stores/{store_name}/documents/{document_id}?api_key=YOUR_API_KEY"

# =====================================================
# CREATE STORE (creates a Gemini File Search store)
# =====================================================
@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    """
    Create file search store on Gemini. Caller must pass their Gemini API key.
    Response contains:
      - store_name: what you provided (local identifier)
      - file_search_store_resource: the Gemini resource name (fileSearchStores/...)
      - created_at, file_count
    Note: no extra instructions are returned per request.
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
    Upload and index files to the given store. Returns:
      - filename, uploaded, indexed (bool), document_resource, document_id, gemini_error
    (No deletion instructions returned in response per your request.)
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

    # temp folder for this upload - cleaned after indexing
    temp_folder = UPLOAD_ROOT / store_name
    temp_folder.mkdir(parents=True, exist_ok=True)

    results = []

    for upload in files:
        original_filename = upload.filename or "file"
        filename = clean_filename(original_filename)

        temp_path = temp_folder / filename

        # ---- write the file locally in streaming chunks (required by SDK) ----
        size = 0
        try:
            async with aiofiles.open(temp_path, "wb") as out_f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > MAX_FILE_BYTES:
                        # abort write and report
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
            # if file exceeded limit we continued above; skip to next file
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

        # ---- upload & index directly into Gemini File Search (this chunks + embeds + indexes) ----
        document_resource = None
        document_id = None
        indexed_ok = False
        gemini_error = None

        try:
            # Use the sanitized filename in the display_name sent to Gemini
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=fs_store_name,
                config={"display_name": filename}
            )
            # Wait for indexing to complete
            op = wait_for_operation(client, op)

            # Many SDK versions return the newly created document inside operation response.
            try:
                document_resource = op.response.file_search_document.name
            except Exception:
                document_resource = None

            # If SDK did not provide document resource, call REST to list documents and match by displayName
            if not document_resource:
                docs = rest_list_documents_for_store(fs_store_name, api_key)
                # look for document whose displayName matches sanitized filename (best-effort)
                for d in docs:
                    display = d.get("displayName") or d.get("display_name") or ""
                    name = d.get("name") or ""
                    if display and display == filename:
                        document_resource = name
                        break
                # fallback: match partial token
                if not document_resource:
                    for d in docs:
                        name = d.get("name") or ""
                        if filename in name:
                            document_resource = name
                            break

            if document_resource:
                document_id = document_resource.split("/")[-1]
                indexed_ok = True
            else:
                # If no resource returned but SDK operation finished without raising,
                # indexing most likely occurred — still mark indexed_ok True but doc id unknown.
                indexed_ok = True

        except Exception as e:
            gemini_error = str(e)
            indexed_ok = False

        # OPTION A: delete local temp file immediately to avoid disk usage
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass

        # Record metadata locally (we keep metadata even if doc_id is missing)
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

    # return consolidated result
    return {"success": True, "results": results}

# =====================================================
# LIST STORES (with files metadata)
# =====================================================
@app.get("/stores")
def list_stores(api_key: str):
    """List local stores and files. Validates the provided Gemini API key by trying to init the client."""
    try:
        init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    # return stores as list for readability
    return {"success": True, "stores": list(data["file_stores"].values())}

# =====================================================
# DELETE DOCUMENT (calls Gemini REST to remove doc from File Search store)
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
        return JSONResponse({"success": False, "error": f"REST request failed: {e}"}, status_code=500)

    if resp.status_code not in (200, 204):
        return JSONResponse({"success": False, "error": f"Gemini REST DELETE failed: {resp.text}"}, status_code=resp.status_code)

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

    # Attempt delete on Gemini (best-effort)
    try:
        client = init_gemini_client(api_key)
        client.file_search_stores.delete(name=fs_store, config={"force": True})
    except Exception:
        # swallow exceptions - we still remove local metadata
        pass

    # Delete local temp folder
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
        # Build File Search tool
        file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=fs_store_names))
        system_instruction = payload.system_prompt or (
            "You are a document summarization assistant. ONLY summarize information directly found in provided File Search stores."
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

        # extract grounding metadata if present
        grounding = None
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            grounding = getattr(response.candidates[0], "grounding_metadata", None)

        return {"success": True, "response_text": getattr(response, "text", ""), "grounding_metadata": grounding}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
