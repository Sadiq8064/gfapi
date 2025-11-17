# gemini_file_search_api.py
"""
FastAPI backend upgraded to use REAL Gemini File Search (per-store RAG).
- Each create_store() creates a Gemini File Search store and stores the store resource name locally.
- upload -> upload_to_file_search_store (wait for indexing)
- ask -> uses the File Search tool so Gemini performs vector retrieval.
Notes:
- Clients must pass their Gemini API key on every request (we never persist it).
- This code attempts to follow the official GenAI Python client examples.
"""

import os
import time
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import aiofiles

# Try import of Google GenAI client. If not installed, endpoints that call Gemini will error clearly.
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# -------- CONFIG ----------
DATA_FILE = "gemini_stores.json"
UPLOAD_ROOT = Path("uploads")
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB per-file default (can be disabled via form param)
POLL_INTERVAL = 2  # seconds to poll long-running operations
# --------------------------

app = FastAPI(title="Gemini File Search RAG API (per-store)")

# ----------------- Helpers: persistence -----------------

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

# ----------------- Pydantic models -----------------

class CreateStoreRequest(BaseModel):
    api_key: str
    store_name: str

class AskRequest(BaseModel):
    api_key: str
    stores: List[str]           # list of store names (exact as created)
    question: str
    system_prompt: Optional[str] = None

# ----------------- Gemini client helper -----------------

def init_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai client library is not installed on the server.")
    try:
        # Initialize per request (do NOT persist api_key)
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

# ----------------- Utility: poll long-running operation -----------------

def wait_for_operation(client, operation):
    """
    Poll operation until done. Returns operation result object.
    If operation contains error, raises RuntimeError.
    NOTE: exact structure of operation depends on SDK; we attempt to check `.done` and errors.
    """
    # Many SDKs let you call client.operations.get(operation) with the operation object or name
    op = operation
    try:
        # If operation has attribute 'name' we will use that to fetch fresh state
        op_name = getattr(op, "name", None)
    except Exception:
        op_name = None

    while True:
        done = getattr(op, "done", None)
        if done:
            break
        # try refresh via operations.get if available
        try:
            if op_name and hasattr(client, "operations") and hasattr(client.operations, "get"):
                op = client.operations.get(op)
            else:
                # last-resort: sleep and continue
                time.sleep(POLL_INTERVAL)
                op = op  # nothing else we can do
        except Exception:
            time.sleep(POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)

    # Check for error
    err = getattr(op, "error", None)
    if err:
        raise RuntimeError(f"Operation failed: {err}")
    return op

# ----------------- Endpoint: Create store -----------------

@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    """
    Create a Gemini File Search store (real vector store). Returns minimal info:
    { "success": True, "store_name": ..., "file_search_store_resource": "...", "created_at": "...", "file_count": 0 }
    """
    # initialize gemini client to validate key
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    if payload.store_name in data["file_stores"]:
        return JSONResponse({"error": "A store with this name already exists."}, status_code=400)

    # create File Search store on Gemini
    try:
        fs_store = client.file_search_stores.create(config={"display_name": payload.store_name})
        # fs_store.name is the resource identifier like "fileSearchStores/..."
        fs_store_name = getattr(fs_store, "name", None) or fs_store
    except Exception as e:
        return JSONResponse({"error": f"Failed to create File Search store on Gemini: {e}"}, status_code=500)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # Local metadata: store by the exact user-provided store_name
    meta = {
        "store_name": payload.store_name,
        "file_search_store_name": fs_store_name,
        "created_at": created_at,
        "files": []  # list of dicts: display_name, uploaded_at, size_bytes, gemini_import_op (optional)
    }

    data["file_stores"][payload.store_name] = meta
    data["current_store_name"] = payload.store_name
    save_data(data)

    # create local uploads folder (optional; we will store local copies as well)
    local_folder = UPLOAD_ROOT / payload.store_name
    local_folder.mkdir(parents=True, exist_ok=True)

    return {
        "success": True,
        "store_name": payload.store_name,
        "file_search_store_resource": fs_store_name,
        "created_at": created_at,
        "file_count": 0
    }

# ----------------- Endpoint: Upload files (directly into File Search store) -----------------

@app.post("/stores/{store_name}/upload")
async def upload_files(
    store_name: str,
    api_key: str = Form(...),
    limit: Optional[bool] = Form(True),
    files: List[UploadFile] = File(...)
):
    """
    Upload files directly into the Gemini File Search store using upload_to_file_search_store.
    - api_key: user's Gemini API key
    - limit: if True enforce MAX_FILE_BYTES per-file; if False, skip size check
    - files: multipart files (multiple allowed)
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
        raise HTTPException(status_code=500, detail="File Search store mapping missing for this store")

    local_folder = UPLOAD_ROOT / store_name
    local_folder.mkdir(parents=True, exist_ok=True)

    results = []

    for upload in files:
        filename = upload.filename
        target_path = local_folder / filename

        # stream file locally
        size = 0
        try:
            async with aiofiles.open(target_path, "wb") as out_f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > MAX_FILE_BYTES:
                        await out_f.close()
                        try:
                            os.remove(target_path)
                        except Exception:
                            pass
                        results.append({
                            "filename": filename,
                            "uploaded": False,
                            "reason": f"File exceeds {MAX_FILE_BYTES} bytes (50MB)."
                        })
                        break
                    await out_f.write(chunk)
            # if we added a failure for this filename above, continue
            if results and results[-1].get("filename") == filename and not results[-1].get("uploaded", True):
                continue
        except Exception as e:
            results.append({"filename": filename, "uploaded": False, "reason": f"Failed to save locally: {e}"})
            continue

        # Now upload directly into File Search store (this indexes & chunks & creates embeddings)
        try:
            # Call upload_to_file_search_store with file path and store resource name
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(target_path),
                file_search_store_name=fs_store_name,
                config={"display_name": filename}
            )
            # Wait for import/indexing to complete
            op = wait_for_operation(client, op)
        except Exception as e:
            # indexing failed; keep local file but mark gemini error
            store_meta["files"].append({
                "display_name": filename,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": size,
                "gemini_indexed": False,
                "gemini_error": str(e),
                "operation": getattr(op, "name", None) if 'op' in locals() else None
            })
            save_data(data)
            results.append({"filename": filename, "uploaded": True, "indexed": False, "gemini_error": str(e)})
            continue

        # If op completed successfully, record file metadata locally.
        # The SDK may not return a document name in a uniform way; we record the operation name and mark indexed.
        store_file_entry = {
            "display_name": filename,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "size_bytes": size,
            "gemini_indexed": True,
            "index_operation": getattr(op, "name", None) if op is not None else None
        }
        store_meta["files"].append(store_file_entry)
        save_data(data)

        results.append({
            "filename": filename,
            "uploaded": True,
            "indexed": True,
            "size_bytes": size,
            "uploaded_at": store_file_entry["uploaded_at"]
        })

    # persist final metadata
    data["file_stores"][store_name] = store_meta
    save_data(data)

    return {"success": True, "results": results}

# ----------------- Endpoint: List all stores -----------------

@app.get("/stores")
def list_stores(api_key: str):
    """
    Returns all local stores and file counts & metadata. Validates API key by initializing the client.
    """
    try:
        _ = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    stores = []
    for store_name, meta in data["file_stores"].items():
        files = meta.get("files", [])
        stores.append({
            "store_name": store_name,
            "file_search_store_resource": meta.get("file_search_store_name"),
            "created_at": meta.get("created_at"),
            "file_count": len(files),
            "files": files
        })
    return {"success": True, "stores": stores}

# ----------------- Endpoint: Delete entire store -----------------

@app.delete("/stores/{store_name}")
def delete_store(store_name: str, api_key: str):
    """
    Delete the File Search store on Gemini (force) and remove local metadata & files.
    """
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    meta = data["file_stores"][store_name]
    fs_store_name = meta.get("file_search_store_name")
    deleted_files = []
    errors = []

    # Attempt to delete the File Search store on Gemini
    if fs_store_name:
        try:
            client.file_search_stores.delete(name=fs_store_name, config={"force": True})
        except Exception as e:
            errors.append(f"Failed to delete File Search store on Gemini: {e}")

    # Delete local files
    local_folder = UPLOAD_ROOT / store_name
    if local_folder.exists():
        try:
            shutil.rmtree(local_folder)
        except Exception as e:
            errors.append(f"Failed to remove local folder: {e}")

    # Record deleted file names (local metadata)
    for f in meta.get("files", []):
        deleted_files.append(f.get("display_name"))

    # Remove metadata
    del data["file_stores"][store_name]
    if data.get("current_store_name") == store_name:
        data["current_store_name"] = None
    save_data(data)

    return {"success": True, "deleted_store": store_name, "deleted_files": deleted_files, "errors": errors}

# ----------------- Endpoint: Delete specific upload (best-effort) -----------------

@app.delete("/stores/{store_name}/files/{file_name}")
def delete_upload(store_name: str, file_name: str, api_key: str):
    """
    Attempt to remove a single file from a File Search store.
    NOTE: Depending on the GenAI SDK, deleting a single document from a file search store
    may require a documents API. If supported, we attempt to delete; otherwise we remove local metadata.
    """
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    meta = data["file_stores"][store_name]
    local_file = None
    for f in meta.get("files", []):
        if f.get("display_name") == file_name:
            local_file = f
            break
    if not local_file:
        raise HTTPException(status_code=404, detail="File not found in store metadata")

    errors = []
    result = {"display_name": file_name, "removed_from_local": False, "removed_from_gemini": False}

    # Try to delete from File Search store using any available document deletion method.
    # The SDK example does not show a direct delete document method; some SDKs expose a documents API.
    try:
        # Try an assumed method name first (may need adjustment for SDK)
        if hasattr(client.file_search_stores, "delete_document"):
            # if implemented: client.file_search_stores.delete_document(name=doc_resource_name)
            # We do not always have doc resource name; check if we stored it earlier.
            doc_resource = local_file.get("document_resource") or local_file.get("document_name")
            if doc_resource:
                client.file_search_stores.delete_document(name=doc_resource)
                result["removed_from_gemini"] = True
            else:
                # No document resource recorded; cannot delete specific doc server-side reliably.
                errors.append("No document resource recorded; cannot delete specific document on Gemini.")
        else:
            errors.append("SDK does not implement file_search_stores.delete_document; skipping server-side deletion.")
    except Exception as e:
        errors.append(f"Error deleting document from Gemini: {e}")

    # Delete local file copy if exists
    try:
        local_path = local_file.get("local_path") or (UPLOAD_ROOT / store_name / file_name)
        if os.path.exists(str(local_path)):
            os.remove(str(local_path))
            result["removed_from_local"] = True
    except Exception as e:
        errors.append(f"Failed to delete local file: {e}")

    # Remove metadata entry locally
    try:
        meta["files"] = [f for f in meta.get("files", []) if f.get("display_name") != file_name]
        data["file_stores"][store_name] = meta
        save_data(data)
    except Exception as e:
        errors.append(f"Failed to update metadata: {e}")

    return {"success": True, "result": result, "errors": errors}

# ----------------- Endpoint: Storage info -----------------

@app.get("/storage")
def storage_info(api_key: str):
    try:
        _ = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    total_bytes = 0
    stores = []
    for name, meta in data["file_stores"].items():
        store_size = sum(f.get("size_bytes", 0) for f in meta.get("files", []))
        stores.append({
            "store_name": name,
            "file_count": len(meta.get("files", [])),
            "size_bytes": store_size,
            "size_mb": round(store_size / (1024 * 1024), 3),
            "file_search_store_resource": meta.get("file_search_store_name")
        })
        total_bytes += store_size

    return {"success": True, "total_size_bytes": total_bytes, "total_size_mb": round(total_bytes / (1024 * 1024), 3), "stores": stores}

# ----------------- Endpoint: Ask question using File Search tool (RAG) -----------------

@app.post("/ask")
def ask_question(payload: AskRequest):
    """
    Perform vector retrieval via File Search stores and ask the model.
    Payload:
    {
      "api_key": "...",
      "stores": ["store1", "store2"],
      "question": "Your question",
      "system_prompt": "optional system instruction"
    }
    Returns: response_text, raw_response, grounding_metadata (if present)
    """
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()

    # collect the file_search_store resource names for the requested stores
    fs_store_names = []
    for s in payload.stores:
        if s in data["file_stores"]:
            v = data["file_stores"][s].get("file_search_store_name")
            if v:
                fs_store_names.append(v)

    if not fs_store_names:
        return JSONResponse({"error": "No valid File Search stores found for the provided store names."}, status_code=400)

    # build tool config for File Search
    # Using types.Tool -> types.FileSearch example from SDK docs
    try:
        file_search_tool = types.Tool(
            file_search=types.FileSearch(file_search_store_names=fs_store_names)
        )
        system_instruction = payload.system_prompt or (
            "You are a document summarization assistant. ONLY summarize information directly found in provided File Search stores. "
            "If nothing relevant is found return: 'No relevant information found in the documents'."
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
    except Exception as e:
        return JSONResponse({"error": f"generate_content/file_search error: {e}"}, status_code=500)

    # extract response text and grounding metadata (citations)
    response_text = getattr(response, "text", "") or ""
    grounding = None
    try:
        # candidates -> grounding_metadata
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            cand = response.candidates[0]
            grounding = getattr(cand, "grounding_metadata", None) or getattr(cand, "groundingMetadata", None)
    except Exception:
        grounding = None

    return {"success": True, "response_text": response_text, "grounding_metadata": grounding, "raw_response": repr(response)}

# ----------------- Run instruction --------------
# Run app with: uvicorn gemini_file_search_api:app --host 0.0.0.0 --port 8000
