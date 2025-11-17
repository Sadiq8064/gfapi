# gemini_file_search_api.py
import os
import io
import time
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
from pathlib import Path

# NOTE: the google genai client package name may vary. The example below matches
# the usage pattern you used in the original script:
#   from google import genai
#   from google.genai import types
#
# If your installed package is named differently, change imports accordingly.
try:
    from google import genai
    from google.genai import types
except Exception as e:
    genai = None
    types = None
    # We'll still let server run, endpoints that call Gemini will return an error.

# Config
DATA_FILE = "gemini_stores.json"
UPLOAD_ROOT = Path("uploads")
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

app = FastAPI(title="Gemini File Search API")

# --- Helpers: persistence for stores ---
def _ensure_dirs():
    UPLOAD_ROOT.mkdir(exist_ok=True, parents=True)

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"file_stores": {}, "current_store_name": None}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def create_store_metadata(display_name: str):
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    internal_name = f"store-{display_name.lower().replace(' ', '-')}"
    return {
        "name": internal_name,
        "display_name": display_name,
        "created_at": created_at,
        "files": []  # list of file metadata dicts
    }

def init_storage():
    _ensure_dirs()
    if not os.path.exists(DATA_FILE):
        save_data({"file_stores": {}, "current_store_name": None})

init_storage()

# --- Pydantic models for request bodies ---
class CreateStoreRequest(BaseModel):
    api_key: str
    store_name: str

class BasicApiKey(BaseModel):
    api_key: str

class AskRequest(BaseModel):
    api_key: str
    stores: List[str]
    question: str
    system_prompt: Optional[str] = None

# --- Utility: initialize gemini client per-request ---
def init_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai client library not installed on server.")
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

# --- Endpoint: Create store ---
@app.post("/stores/create")
def create_store(request: CreateStoreRequest):
    """Create a new store. Caller must pass their Gemini API key and store name."""
    # We only use the api_key to validate the user provided one — we will initialise client to ensure key is valid.
    try:
        _ = init_gemini_client(request.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    if request.store_name in data["file_stores"]:
        return JSONResponse({"error": "A store with this name already exists."}, status_code=400)

    meta = create_store_metadata(request.store_name)
    data["file_stores"][request.store_name] = meta
    data["current_store_name"] = request.store_name
    save_data(data)

    # Make uploads folder for store
    store_folder = UPLOAD_ROOT / meta["name"]
    store_folder.mkdir(parents=True, exist_ok=True)

    return {
    "success": True,
    "store_name": request.store_name,
    "created_at": meta["created_at"],
    "file_count": 0
}


# --- Endpoint: Upload files to a store ---
@app.post("/stores/{store_name}/upload")
async def upload_files(
    store_name: str,
    api_key: str = Form(...),
    limit: Optional[bool] = Form(True),
    files: List[UploadFile] = File(...)
):
    """
    Upload one or more files to the named store.
    - api_key: user's Gemini API key (required)
    - limit (form boolean): if True (default) enforce 50 MB limit per-file, else allow larger files
    - files: multipart file uploads (can be multiple)
    """
    # Validate store exists
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    store_meta = data["file_stores"][store_name]
    store_folder = UPLOAD_ROOT / store_meta["name"]
    store_folder.mkdir(parents=True, exist_ok=True)

    uploaded_results = []
    for upload in files:
        filename = os.path.basename(upload.filename)
        # stream to disk and measure size without loading all into memory
        target_path = store_folder / filename

        size = 0
        # write streaming
        try:
            async with aiofiles.open(target_path, "wb") as out_file:
                while True:
                    chunk = await upload.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > MAX_FILE_BYTES:
                        # abort write and remove partial file
                        await out_file.close()
                        try:
                            os.remove(target_path)
                        except Exception:
                            pass
                        uploaded_results.append({
                            "filename": filename,
                            "uploaded": False,
                            "reason": f"File exceeds limit of {MAX_FILE_BYTES} bytes (50MB)."
                        })
                        # consume remaining stream (UploadFile may be read fully already)
                        break
                    await out_file.write(chunk)
            # If we aborted due to size, skip further processing
            if uploaded_results and uploaded_results[-1].get("filename") == filename and not uploaded_results[-1]["uploaded"]:
                continue
        except Exception as e:
            uploaded_results.append({
                "filename": filename,
                "uploaded": False,
                "reason": f"Failed to save locally: {e}"
            })
            continue

        # Attempt to upload to Gemini Files API (best-effort). If it fails, keep local copy and return metadata.
        gemini_file_info = {}
        try:
            # genai .files.upload may accept path or file-like; using path as many SDKs accept it
            uploaded_file = client.files.upload(file=str(target_path))
            # SDKs differ; attempt to extract typical fields
            gemini_file_info = {
                "file_id": getattr(uploaded_file, "name", None) or getattr(uploaded_file, "id", None),
                "uri": getattr(uploaded_file, "uri", None),
                "mime_type": getattr(uploaded_file, "mime_type", None)
            }
        except Exception as e:
            # do not fail the whole request; just note the gemini upload error
            gemini_file_info = {"error": f"Gemini upload failed: {e}"}

        # record metadata
        file_meta = {
            "file_id": gemini_file_info.get("file_id"),
            "display_name": filename,
            "file_path": str(target_path),
            "size_bytes": os.path.getsize(target_path),
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_type": os.path.splitext(filename)[1].lower(),
            "gemini_file": gemini_file_info
        }

        store_meta["files"].append(file_meta)
        uploaded_results.append({
            "filename": filename,
            "uploaded": True,
            "size_bytes": file_meta["size_bytes"],
            "upload_time": file_meta["upload_time"],
            "gemini_file": gemini_file_info
        })

    # persist
    data["file_stores"][store_name] = store_meta
    save_data(data)

    return {"success": True, "uploaded": uploaded_results}

# --- Endpoint: List all stores and their uploads ---
@app.get("/stores")
def list_stores(api_key: str):
    """
    List all stores and metadata.
    The api_key is required by your spec but is not stored.
    """
    # Validate API key by init to ensure it's correct (optional)
    try:
        _ = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    stores = []
    for display_name, meta in data.get("file_stores", {}).items():
        files = []
        for f in meta.get("files", []):
            files.append({
                "display_name": f.get("display_name"),
                "size_bytes": f.get("size_bytes"),
                "upload_time": f.get("upload_time"),
                "file_type": f.get("file_type"),
                "file_id": f.get("file_id"),
                "gemini_file": f.get("gemini_file")
            })
        stores.append({
            "display_name": meta.get("display_name"),
            "name": meta.get("name"),
            "created_at": meta.get("created_at"),
            "files_count": len(files),
            "files": files
        })
    return {"success": True, "stores": stores}

# --- Endpoint: Delete a complete store ---
@app.delete("/stores/{store_name}")
def delete_store(store_name: str, api_key: str):
    """
    Delete entire store and its uploads (both local and, if possible, on Gemini).
    Returns metadata about what was deleted and whether deletion succeeded.
    """
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    store_meta = data["file_stores"][store_name]
    deleted_files = []
    errors = []

    # delete each file on Gemini if file_id present
    for f in list(store_meta.get("files", [])):
        file_id = f.get("file_id")
        fname = f.get("display_name")
        deleted_entry = {"filename": fname}
        if file_id:
            try:
                client.files.delete(name=file_id)
                deleted_entry["gemini_deleted"] = True
            except Exception as e:
                deleted_entry["gemini_deleted"] = False
                deleted_entry["gemini_error"] = str(e)
                errors.append(f"Gemini delete failed for {fname}: {e}")
        else:
            deleted_entry["gemini_deleted"] = False
            deleted_entry["gemini_error"] = "No gemini file id recorded"

        # delete local file
        try:
            local_path = f.get("file_path")
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
                deleted_entry["local_deleted"] = True
            else:
                deleted_entry["local_deleted"] = False
        except Exception as e:
            deleted_entry["local_deleted"] = False
            deleted_entry["local_error"] = str(e)
            errors.append(f"Local delete failed for {fname}: {e}")

        deleted_files.append(deleted_entry)

    # remove store folder
    try:
        folder = UPLOAD_ROOT / store_meta["name"]
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)
    except Exception as e:
        errors.append(f"Failed to remove store folder: {e}")

    # Remove metadata
    del data["file_stores"][store_name]
    if data.get("current_store_name") == store_name:
        data["current_store_name"] = None
    save_data(data)

    return {"success": True, "deleted_store": store_name, "deleted_files": deleted_files, "errors": errors}

# --- Endpoint: Delete specific upload ---
@app.delete("/stores/{store_name}/files/{file_display_name}")
def delete_upload(store_name: str, file_display_name: str, api_key: str):
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")
    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    store_meta = data["file_stores"][store_name]
    target_idx = None
    for idx, f in enumerate(store_meta.get("files", [])):
        if f.get("display_name") == file_display_name:
            target_idx = idx
            break

    if target_idx is None:
        raise HTTPException(status_code=404, detail="File not found in store")

    file_meta = store_meta["files"].pop(target_idx)
    errors = []
    resp = {"filename": file_meta.get("display_name")}

    # Delete on Gemini if we have ID
    file_id = file_meta.get("file_id")
    if file_id:
        try:
            client.files.delete(name=file_id)
            resp["gemini_deleted"] = True
        except Exception as e:
            resp["gemini_deleted"] = False
            resp["gemini_error"] = str(e)
            errors.append(str(e))
    else:
        resp["gemini_deleted"] = False
        resp["gemini_error"] = "No gemini file id recorded"

    # Delete local file
    try:
        local_path = file_meta.get("file_path")
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
            resp["local_deleted"] = True
        else:
            resp["local_deleted"] = False
    except Exception as e:
        resp["local_deleted"] = False
        resp["local_error"] = str(e)
        errors.append(str(e))

    data["file_stores"][store_name] = store_meta
    save_data(data)

    return {"success": True, "result": resp, "errors": errors}

# --- Endpoint: Check storage usage ---
@app.get("/storage")
def check_storage(api_key: str):
    """
    Returns storage usage: total MB used, stores created, and per-store storage usage.
    """
    # Optionally check api_key validity
    try:
        _ = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    total_bytes = 0
    stores = []
    for display_name, meta in data.get("file_stores", {}).items():
        store_bytes = 0
        for f in meta.get("files", []):
            store_bytes += int(f.get("size_bytes", 0) or 0)
        stores.append({
            "display_name": meta.get("display_name"),
            "name": meta.get("name"),
            "files_count": len(meta.get("files", [])),
            "size_bytes": store_bytes,
            "size_mb": round(store_bytes / (1024 * 1024), 3)
        })
        total_bytes += store_bytes

    return {
        "success": True,
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 3),
        "stores": stores
    }

# --- Endpoint: Ask question against stores (RAG) ---
@app.post("/ask")
def ask_question(payload: AskRequest):
    """
    Question-answering endpoint. Caller must pass:
      - api_key (string): their Gemini API key
      - stores (list[str]): store display names to search (one or many)
      - question (string)
      - system_prompt (optional string) to override default behavior
    Returns:
      - summarized response text (response_text)
      - raw response object (as repr)
      - list of doc URIs used (raw citations if any)
    """
    # Validate client
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    # collect parts
    contents = []

    for store_name in payload.stores:
        if store_name not in data["file_stores"]:
            # skip unknown stores (or optionally return error)
            continue
        for f in data["file_stores"][store_name].get("files", []):
            gemini_info = f.get("gemini_file") or {}
            # Prefer to use gemini URI if available
            uri = gemini_info.get("uri")
            mime_type = gemini_info.get("mime_type") or f.get("file_type")
            if uri:
                try:
                    contents.append(types.Part.from_uri(file_uri=uri, mime_type=mime_type))
                except Exception:
                    # fallback to path
                    if os.path.exists(f.get("file_path", "")):
                        try:
                            contents.append(types.Part.from_path(path=f.get("file_path")))
                        except Exception:
                            # give up for this file
                            pass
            else:
                # use local path if present
                local_path = f.get("file_path")
                if local_path and os.path.exists(local_path):
                    try:
                        contents.append(types.Part.from_path(path=local_path))
                    except Exception:
                        pass

    # append the question text at the end (Gemini SDK usually expects text part)
    contents.append(payload.question)

    # system prompt: use provided system_prompt else fallback to a safe default
    system_instruction = payload.system_prompt or (
        "You are a document summarization assistant. ONLY summarize information directly found in provided documents. "
        "If nothing relevant is found return: \"No relevant information found in the documents\"."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2)
        )
    except Exception as e:
        return JSONResponse({"error": f"Gemini generate_content error: {e}"}, status_code=500)

    # Attempt to extract a reasonable response text and citations
    response_text = None
    raw = None
    citations = []

    # Many SDKs provide .text or .content; attempt to extract gracefully
    try:
        response_text = getattr(response, "text", None)
        if response_text is None:
            # Try other possible attributes
            if hasattr(response, "candidates") and len(response.candidates) > 0:
                response_text = getattr(response.candidates[0], "output", None) or getattr(response.candidates[0], "content", None)
    except Exception:
        response_text = None

    # Raw repr
    try:
        raw = repr(response)
    except Exception:
        raw = str(response)

    # Try to extract citations if provided by SDK in response.metadata or similar
    try:
        # example: response.citations or response.metadata; attempt a couple of options
        if hasattr(response, "metadata"):
            metadata = getattr(response, "metadata")
            if metadata:
                citations.append(metadata)
        if hasattr(response, "citations"):
            citations.extend(getattr(response, "citations") or [])
        # some SDKs embed sources inside candidates
        if hasattr(response, "candidates"):
            for c in response.candidates:
                if hasattr(c, "citation_metadata"):
                    citations.append(getattr(c, "citation_metadata"))
    except Exception:
        pass

    return {
        "success": True,
        "response_text": response_text or "",
        "raw_response": raw,
        "citations": citations
    }

# Run via: uvicorn gemini_file_search_api:app --host 0.0.0.0 --port 8000
