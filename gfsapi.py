# ==========================================================
# Gemini File Search RAG API (Production Ready)
# Option A: Delete local file immediately after upload/index
# Uses REAL Google File Search (vector indexing)
# ==========================================================

import os
import time
import json
import shutil
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import aiofiles

# Google Gemini SDK
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------- CONFIG ----------------
DATA_FILE = "gemini_stores.json"
UPLOAD_ROOT = Path("uploads")  # temp save for upload
MAX_FILE_BYTES = 50 * 1024 * 1024
POLL_INTERVAL = 2
# ----------------------------------------

app = FastAPI(title="Gemini File Search RAG API – Clean, Production Ready")

# ---------------- Helpers ----------------

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

# ----------------------------------------

class CreateStoreRequest(BaseModel):
    api_key: str
    store_name: str

class AskRequest(BaseModel):
    api_key: str
    stores: List[str]
    question: str
    system_prompt: Optional[str] = None

# ----------------------------------------

def init_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai library missing.")
    return genai.Client(api_key=api_key)

# ----------------------------------------

def wait_for_operation(client, operation):
    """Polls for file indexing completion."""
    op = operation
    name = getattr(op, "name", None)

    while not getattr(op, "done", False):
        time.sleep(POLL_INTERVAL)
        try:
            op = client.operations.get(op)
        except Exception:
            pass

    if getattr(op, "error", None):
        raise RuntimeError(op.error)

    return op

# =====================================================
# CREATE STORE  (REAL FILE SEARCH STORE)
# =====================================================
@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()

    if payload.store_name in data["file_stores"]:
        return JSONResponse({"error": "Store name already exists."}, status_code=400)

    try:
        store = client.file_search_stores.create(config={"display_name": payload.store_name})
        fs_store = store.name
    except Exception as e:
        return JSONResponse({"error": f"Error creating File Search store: {e}"}, status_code=500)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    data["file_stores"][payload.store_name] = {
        "store_name": payload.store_name,
        "file_search_store_name": fs_store,
        "created_at": created_at,
        "files": []
    }

    data["current_store_name"] = payload.store_name
    save_data(data)

    return {
        "success": True,
        "store_name": payload.store_name,
        "file_search_store_resource": fs_store,
        "created_at": created_at,
        "file_count": 0
    }

# =====================================================
# UPLOAD (INDEX) FILE → DELETE LOCAL AFTER SUCCESS
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
    fs_store = store_meta["file_search_store_name"]

    local_folder = UPLOAD_ROOT / store_name
    local_folder.mkdir(parents=True, exist_ok=True)

    results = []

    for upload in files:
        filename = upload.filename
        temp_path = local_folder / filename

        # Write locally (required by SDK)
        size = 0
        try:
            async with aiofiles.open(temp_path, "wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > MAX_FILE_BYTES:
                        await f.close()
                        os.remove(temp_path)
                        results.append({
                            "filename": filename,
                            "uploaded": False,
                            "reason": "File exceeds 50MB limit"
                        })
                        continue
                    await f.write(chunk)
        except Exception as e:
            results.append({"filename": filename, "uploaded": False, "reason": str(e)})
            continue

        # Upload → Index
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=fs_store,
                config={"display_name": filename}
            )
            op = wait_for_operation(client, op)

            # Extract document name if returned
            doc_name = None
            try:
                doc_name = op.response.file_search_document.name
            except:
                doc_name = None

            # OPTION A: DELETE file immediately
            try:
                os.remove(temp_path)
            except:
                pass

            # Record metadata
            entry = {
                "display_name": filename,
                "size_bytes": size,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gemini_indexed": True,
                "document_resource": doc_name,
                "document_id": doc_name.split("/")[-1] if doc_name else None
            }

            store_meta["files"].append(entry)
            save_data(data)

            results.append({
                "filename": filename,
                "uploaded": True,
                "indexed": True,
                "document_resource": doc_name
            })

        except Exception as e:
            results.append({"filename": filename, "uploaded": True, "indexed": False, "error": str(e)})

    save_data(data)
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
# DELETE DOCUMENT FROM STORE (REAL REST API CALL)
# =====================================================
@app.delete("/stores/{store_name}/documents/{document_id}")
def delete_document(store_name: str, document_id: str, api_key: str):
    data = load_data()

    if store_name not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store_name]
    fs_store = meta["file_search_store_name"]

    doc_resource = f"{fs_store}/documents/{document_id}"

    # Call REST API directly
    url = f"https://generativelanguage.googleapis.com/v1beta/{doc_resource}?force=true&key={api_key}"

    res = requests.delete(url)

    if res.status_code not in (200, 204):
        return JSONResponse({
            "success": False,
            "error": f"Gemini REST DELETE failed: {res.text}"
        }, status_code=res.status_code)

    # Remove from local metadata
    meta["files"] = [
        f for f in meta["files"]
        if f.get("document_id") != document_id
    ]

    data["file_stores"][store_name] = meta
    save_data(data)

    return {"success": True, "deleted": document_id}

# =====================================================
# DELETE STORE (LOCAL + REMOTE)
# =====================================================
@app.delete("/stores/{store_name}")
def delete_store(store_name: str, api_key: str):
    data = load_data()

    if store_name not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store_name]
    fs_store = meta["file_search_store_name"]

    try:
        client = init_gemini_client(api_key)
        client.file_search_stores.delete(name=fs_store, config={"force": True})
    except Exception as e:
        pass

    # Delete local temp folder
    folder = UPLOAD_ROOT / store_name
    if folder.exists():
        shutil.rmtree(folder)

    del data["file_stores"][store_name]
    save_data(data)

    return {"success": True, "deleted_store": store_name}

# =====================================================
# ASK QUESTION (REAL RAG)
# =====================================================
@app.post("/ask")
def ask_question(payload: AskRequest):
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()

    fs_stores = []
    for s in payload.stores:
        if s in data["file_stores"]:
            fs_stores.append(data["file_stores"][s]["file_search_store_name"])

    if not fs_stores:
        return JSONResponse({"error": "No valid stores"}, status_code=400)

    try:
        tool = types.Tool(
            file_search=types.FileSearch(file_search_store_names=fs_stores)
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=payload.question,
            config=types.GenerateContentConfig(
                tools=[tool],
                system_instruction=payload.system_prompt or
                "Summarize only information found in the File Search documents.",
                temperature=0.2
            )
        )

        grounding = None
        if hasattr(response, "candidates"):
            grounding = getattr(response.candidates[0], "grounding_metadata", None)

        return {
            "success": True,
            "response_text": response.text,
            "grounding_metadata": grounding
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
