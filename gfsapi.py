# gemini_file_search_api.py
import os
import io
import time
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

DATA_FILE = "gemini_stores.json"
UPLOAD_ROOT = Path("uploads")
MAX_FILE_BYTES = 50 * 1024 * 1024

app = FastAPI(title="Gemini File Search API")

# ------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------

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

def init_storage():
    _ensure_dirs()
    if not os.path.exists(DATA_FILE):
        save_data({"file_stores": {}, "current_store_name": None})

init_storage()

class CreateStoreRequest(BaseModel):
    api_key: str
    store_name: str

class AskRequest(BaseModel):
    api_key: str
    stores: List[str]
    question: str
    system_prompt: Optional[str] = None

# ------------------------------------------------------
# Initialize Gemini Client
# ------------------------------------------------------

def init_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("Gemini client library not installed.")
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

# ------------------------------------------------------
# CREATE STORE  (UPDATED)
# ------------------------------------------------------

@app.post("/stores/create")
def create_store(request: CreateStoreRequest):

    try:
        _ = init_gemini_client(request.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()

    if request.store_name in data["file_stores"]:
        return JSONResponse({"error": "A store with this name already exists."}, status_code=400)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # Exact store name used as folder & key
    meta = {
        "name": request.store_name,
        "display_name": request.store_name,
        "created_at": created_at,
        "files": []
    }

    data["file_stores"][request.store_name] = meta
    data["current_store_name"] = request.store_name
    save_data(data)

    # Create store folder
    store_folder = UPLOAD_ROOT / request.store_name
    store_folder.mkdir(parents=True, exist_ok=True)

    return {
        "success": True,
        "store_name": request.store_name,
        "created_at": created_at,
        "file_count": 0
    }

# ------------------------------------------------------
# UPLOAD FILES
# ------------------------------------------------------

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
    store_folder = UPLOAD_ROOT / store_name
    store_folder.mkdir(parents=True, exist_ok=True)

    uploaded_results = []

    for upload in files:
        filename = os.path.basename(upload.filename)
        target_path = store_folder / filename
        size = 0

        try:
            async with aiofiles.open(target_path, "wb") as out_file:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > MAX_FILE_BYTES:
                        await out_file.close()
                        os.remove(target_path)
                        uploaded_results.append({
                            "filename": filename,
                            "uploaded": False,
                            "reason": "File too large (>50MB)."
                        })
                        break
                    await out_file.write(chunk)

            if uploaded_results and uploaded_results[-1].get("filename") == filename and not uploaded_results[-1]["uploaded"]:
                continue

        except Exception as e:
            uploaded_results.append({
                "filename": filename,
                "uploaded": False,
                "reason": f"Failed to save locally: {e}"
            })
            continue

        gemini_file_info = {}
        try:
            uploaded_file = client.files.upload(file=str(target_path))
            gemini_file_info = {
                "file_id": getattr(uploaded_file, "name", None),
                "uri": getattr(uploaded_file, "uri", None),
                "mime_type": getattr(uploaded_file, "mime_type", None)
            }
        except Exception as e:
            gemini_file_info = {"error": f"Gemini upload failed: {e}"}

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

    data["file_stores"][store_name] = store_meta
    save_data(data)

    return {"success": True, "uploaded": uploaded_results}

# ------------------------------------------------------
# LIST STORES  (SHOW ALL)
# ------------------------------------------------------

@app.get("/stores")
def list_stores(api_key: str):
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
            "created_at": meta["created_at"],
            "file_count": len(files),
            "files": files
        })

    return {"success": True, "stores": stores}

# ------------------------------------------------------
# DELETE STORE
# ------------------------------------------------------

@app.delete("/stores/{store_name}")
def delete_store(store_name: str, api_key: str):
    data = load_data()

    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        client = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    store_meta = data["file_stores"][store_name]
    deleted_files = []

    for f in store_meta.get("files", []):
        entry = {"filename": f["display_name"]}

        try:
            if f.get("file_id"):
                client.files.delete(name=f["file_id"])
                entry["gemini_deleted"] = True
        except Exception as e:
            entry["gemini_deleted"] = False
            entry["gemini_error"] = str(e)

        try:
            if os.path.exists(f["file_path"]):
                os.remove(f["file_path"])
                entry["local_deleted"] = True
        except Exception as e:
            entry["local_deleted"] = False
            entry["local_error"] = str(e)

        deleted_files.append(entry)

    folder = UPLOAD_ROOT / store_name
    if folder.exists():
        shutil.rmtree(folder)

    del data["file_stores"][store_name]
    save_data(data)

    return {"success": True, "deleted_store": store_name, "deleted_files": deleted_files}

# ------------------------------------------------------
# DELETE SPECIFIC FILE
# ------------------------------------------------------

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
    target_file = None

    for f in store_meta["files"]:
        if f["display_name"] == file_display_name:
            target_file = f
            break

    if not target_file:
        raise HTTPException(status_code=404, detail="File not found")

    result = {}

    try:
        if target_file.get("file_id"):
            client.files.delete(name=target_file["file_id"])
            result["gemini_deleted"] = True
    except Exception as e:
        result["gemini_deleted"] = False
        result["gemini_error"] = str(e)

    try:
        if os.path.exists(target_file["file_path"]):
            os.remove(target_file["file_path"])
            result["local_deleted"] = True
    except Exception as e:
        result["local_deleted"] = False
        result["local_error"] = str(e)

    store_meta["files"].remove(target_file)
    data["file_stores"][store_name] = store_meta
    save_data(data)

    return {"success": True, "result": result}

# ------------------------------------------------------
# STORAGE INFO
# ------------------------------------------------------

@app.get("/storage")
def check_storage(api_key: str):
    try:
        _ = init_gemini_client(api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()
    total_bytes = 0
    stores_info = []

    for name, meta in data["file_stores"].items():
        store_size = sum(f.get("size_bytes", 0) for f in meta["files"])
        stores_info.append({
            "store_name": name,
            "file_count": len(meta["files"]),
            "size_bytes": store_size,
            "size_mb": round(store_size / (1024 * 1024), 3)
        })
        total_bytes += store_size

    return {
        "success": True,
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 3),
        "stores": stores_info
    }

# ------------------------------------------------------
# ASK QUESTION (RAG)
# ------------------------------------------------------

@app.post("/ask")
def ask_question(payload: AskRequest):
    try:
        client = init_gemini_client(payload.api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = load_data()

    contents = []

    for store_name in payload.stores:
        if store_name not in data["file_stores"]:
            continue

        for f in data["file_stores"][store_name]["files"]:
            gem = f.get("gemini_file", {})
            uri = gem.get("uri")
            mime = gem.get("mime_type") or f.get("file_type")

            if uri:
                try:
                    contents.append(types.Part.from_uri(file_uri=uri, mime_type=mime))
                except:
                    pass
            else:
                try:
                    contents.append(types.Part.from_path(path=f["file_path"]))
                except:
                    pass

    contents.append(payload.question)

    system_instruction = payload.system_prompt or (
        "Only summarize information directly found in the provided documents."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    response_text = getattr(response, "text", "") or ""
    raw = repr(response)

    return {
        "success": True,
        "response_text": response_text,
        "raw_response": raw
    }
