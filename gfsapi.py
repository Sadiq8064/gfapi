

const GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta";

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function detectMimeType(filename, fallback = "application/octet-stream") {
  if (!filename) return fallback;
  const ext = filename.split(".").pop().toLowerCase();
  const map = {
    pdf: "application/pdf",
    txt: "text/plain",
    md: "text/markdown",
    json: "application/json",
    csv: "text/csv",
    tsv: "text/tab-separated-values",
    xml: "application/xml",
    yaml: "text/yaml",
    yml: "text/yaml",
    doc: "application/msword",
    docx:
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    xls: "application/vnd.ms-excel",
    xlsx:
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ppt: "application/vnd.ms-powerpoint",
    pptx:
      "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    png: "image/png",
    gif: "image/gif",
    webp: "image/webp",
    svg: "image/svg+xml",
    js: "text/javascript",
    ts: "application/typescript",
    html: "text/html",
    css: "text/css",
    zip: "application/zip",
  };
  return map[ext] || fallback;
}

function cleanFilename(name) {
  if (!name) return "file";
  let n = String(name).trim();
  n = n.replace(/\s+/g, "_");
  n = n.replace(/^\.+/, "");
  n = n.replace(/[^A-Za-z0-9_\-\.]/g, "_");
  n = n.replace(/__+/g, "_");
  n = n.replace(/\.\.+/g, ".");
  if (n.length > 180) n = n.slice(0, 180);
  if (!n) return "file";
  return n;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const pathname = url.pathname;
    const method = request.method.toUpperCase();

    // KV helpers
    async function load() {
      const raw = await env.RAG.get("stores");
      return raw ? JSON.parse(raw) : { file_stores: {}, current_store_name: null };
    }
    async function save(data) {
      await env.RAG.put("stores", JSON.stringify(data));
    }

    // Helper: list documents via REST
    async function rest_list_documents_for_store(file_search_store_name, apiKey) {
      const u = `${GEMINI_REST_BASE}/${file_search_store_name}/documents?key=${encodeURIComponent(
        apiKey
      )}`;
      try {
        const resp = await fetch(u, { method: "GET" });
        if (!resp.ok) return [];
        const j = await resp.json();
        return j.documents || [];
      } catch {
        return [];
      }
    }

    // Helper: poll operation until done or timeout (returns opJson or null)
    async function pollOperationUntilDone(operationName, apiKey, timeoutMs = 25000, intervalMs = 2000) {
      const start = Date.now();
      while (Date.now() - start < timeoutMs) {
        try {
          const opUrl = `${GEMINI_REST_BASE}/${operationName}?key=${encodeURIComponent(apiKey)}`;
          const opResp = await fetch(opUrl, { method: "GET" });
          if (!opResp.ok) {
            // if non-200, still retry until timeout
            await new Promise((r) => setTimeout(r, intervalMs));
            continue;
          }
          const opJson = await opResp.json();
          if (opJson.done) return opJson;
        } catch (e) {
          // continue retrying
        }
        await new Promise((r) => setTimeout(r, intervalMs));
      }
      return null;
    }

    // ---------------- ROUTES ----------------

    // CREATE STORE
    if (pathname === "/stores/create" && method === "POST") {
      try {
        const body = await request.json();
        const apiKey = body.api_key || env.GEMINI_API_KEY;
        const storeName = body.store_name;
        if (!apiKey || !storeName) return json({ error: "Missing api_key or store_name" }, 400);

        // Call REST create store
        const createUrl = `${GEMINI_REST_BASE}/fileSearchStores?key=${encodeURIComponent(apiKey)}`;
        const resp = await fetch(createUrl, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ displayName: storeName }),
        });
        if (!resp.ok) {
          const t = await resp.text();
          return json({ error: `Failed to create store: ${t}` }, resp.status);
        }
        const j = await resp.json();
        const fsStoreName = j.name || j; // e.g. fileSearchStores/xxx

        const data = await load();
        if (data.file_stores[storeName]) {
          return json({ error: "Store already exists" }, 400);
        }

        data.file_stores[storeName] = {
          store_name: storeName,
          file_search_store_name: fsStoreName,
          created_at: new Date().toISOString(),
          files: [],
        };
        data.current_store_name = storeName;
        await save(data);

        return json({
          success: true,
          store_name: storeName,
          file_search_store_resource: fsStoreName,
          created_at: data.file_stores[storeName].created_at,
          file_count: 0,
        });
      } catch (err) {
        return json({ error: err.toString() }, 500);
      }
    }

    // UPLOAD FILE(S)
    if (pathname.startsWith("/stores/") && pathname.endsWith("/upload") && method === "POST") {
      const segments = pathname.split("/");
      const storeName = segments[2];
      const data = await load();
      const store = data.file_stores[storeName];
      if (!store) return json({ error: "Store not found" }, 404);

      // Accept api_key either in form or fallback to env
      const form = await request.formData();
      const apiKey = form.get("api_key") || env.GEMINI_API_KEY;
      if (!apiKey) return json({ error: "Missing api_key" }, 400);

      // Collect files - support both "files" key (multiple) and any file entries
      const files = [];
      // formData.getAll('files') may not work if the client names differently; iterate entries
      for (const entry of form.entries()) {
        const [k, v] = entry;
        if (v instanceof File) {
          files.push(v);
        }
      }
      if (files.length === 0) {
        return json({ error: "No files provided in the form-data (file fields required)" }, 400);
      }

      const fsStoreName = store.file_search_store_name;
      const results = [];

      for (const file of files) {
        const origName = file.name || "file";
        const cleanedName = cleanFilename(origName);
        const mimeType = detectMimeType(cleanedName);
        let arrayBuffer;
        try {
          arrayBuffer = await file.arrayBuffer();
        } catch (e) {
          results.push({
            filename: cleanedName,
            uploaded: false,
            indexed: false,
            gemini_error: "Failed to read file contents: " + e.toString(),
          });
          continue;
        }

        // Build multipart with required 'metadata' JSON part and 'file' part
        const fd = new FormData();
        // metadata part must be JSON with displayName and optionally mimeType
        const metadata = { displayName: cleanedName, mimeType };
        fd.append("metadata", new Blob([JSON.stringify(metadata)], { type: "application/json" }));
        fd.append("file", new Blob([arrayBuffer], { type: mimeType }), cleanedName);

        const uploadUrl = `${GEMINI_REST_BASE}/${fsStoreName}:uploadToFileSearchStore?key=${encodeURIComponent(apiKey)}`;

        let opName = null;
        try {
          const uploadResp = await fetch(uploadUrl, {
            method: "POST",
            body: fd,
          });

          if (!uploadResp.ok) {
            const t = await uploadResp.text();
            results.push({
              filename: cleanedName,
              uploaded: false,
              indexed: false,
              gemini_error: t,
            });
            continue;
          }

          // This endpoint may return either an operation name or an immediate response
          const uploadJson = await uploadResp.json().catch((e) => {
            // JSON parse failed
            return null;
          });

          if (!uploadJson) {
            results.push({
              filename: cleanedName,
              uploaded: false,
              indexed: false,
              gemini_error: "Invalid JSON response from Gemini upload endpoint",
            });
            continue;
          }

          // If operation name present, we'll poll in background
          opName = uploadJson.name || uploadJson.operation || null;

          // Some endpoints could return the fileSearchDocument immediately (rare). If so, update now.
          const directDocResource =
            uploadJson?.response?.fileSearchDocument?.name ||
            uploadJson?.fileSearchDocument?.name ||
            null;

          // Save initial metadata (indexed false by default)
          const entry = {
            display_name: cleanedName,
            size_bytes: arrayBuffer.byteLength,
            uploaded_at: new Date().toISOString(),
            gemini_indexed: directDocResource ? true : false,
            document_resource: directDocResource,
            document_id: directDocResource ? directDocResource.split("/").pop() : null,
            gemini_error: null,
            operation_name: opName,
          };

          store.files.push(entry);
          await save(data);

          // If we have an operation, poll in background for up to 25s
          if (opName) {
            ctx.waitUntil((async () => {
              try {
                const opJson = await pollOperationUntilDone(opName, apiKey, 25000, 2000);
                if (opJson && opJson.done) {
                  const docRes =
                    opJson?.response?.fileSearchDocument?.name ||
                    opJson?.response?.file_search_document?.name ||
                    null;
                  const docId = docRes ? docRes.split("/").pop() : null;

                  // Update KV entry: find by operation_name and display_name
                  const refreshed = await load();
                  const localFiles = refreshed.file_stores?.[storeName]?.files || [];
                  for (let f of localFiles) {
                    if (f.operation_name === opName && f.display_name === cleanedName) {
                      f.gemini_indexed = !!docRes;
                      f.document_resource = docRes;
                      f.document_id = docId;
                      delete f.operation_name;
                      break;
                    }
                  }
                  await save(refreshed);
                } else {
                  // timed out - leave operation_name so /sync can find it later
                }
              } catch (bgErr) {
                // swallow background errors
                console.error("Background poll error:", String(bgErr));
              }
            })());
          }

          results.push({
            filename: cleanedName,
            uploaded: true,
            indexed: !!directDocResource,
            document_resource: directDocResource,
            document_id: directDocResource ? directDocResource.split("/").pop() : null,
            gemini_error: null,
            operation_name: opName,
          });
        } catch (e) {
          results.push({
            filename: cleanedName,
            uploaded: false,
            indexed: false,
            gemini_error: e.toString(),
          });
          continue;
        }
      } // end for files

      // Save final store state (some entries were added above)
      await save(data);

      // Immediate response — indexing may continue in background
      return json({ success: true, results });
    }

    // LIST STORES
    if (pathname === "/stores" && method === "GET") {
      try {
        const apiKey = url.searchParams.get("api_key") || env.GEMINI_API_KEY;
        if (!apiKey) return json({ error: "Missing api_key" }, 400);
        // optional: verify key by fetching stores list (lightweight)
        // skip explicit fetch to keep fast; user can detect auth errors on create/upload
        const data = await load();
        return json({ success: true, stores: Object.values(data.file_stores) });
      } catch (e) {
        return json({ error: e.toString() }, 500);
      }
    }

    // SYNC endpoint (populate missing document_ids by listing remote docs)
    if (pathname.startsWith("/stores/") && pathname.endsWith("/sync") && method === "POST") {
      const segments = pathname.split("/");
      const storeName = segments[2];
      const body = await request.json().catch(() => ({}));
      const apiKey = body?.api_key || env.GEMINI_API_KEY;
      if (!apiKey) return json({ error: "Missing api_key" }, 400);

      const data = await load();
      const store = data.file_stores[storeName];
      if (!store) return json({ error: "Store not found" }, 404);

      const fsStore = store.file_search_store_name;
      const docs = await rest_list_documents_for_store(fsStore, apiKey);

      let updated = 0;
      for (const d of docs) {
        const display = d.displayName || d.display_name || "";
        const name = d.name || "";
        if (!display || !name) continue;

        // find matching local entry
        const local = store.files.find(
          (f) =>
            f.display_name === display ||
            (f.display_name && name.includes(f.display_name))
        );
        if (local && !local.document_id) {
          local.document_resource = name;
          local.document_id = name.split("/").pop();
          local.gemini_indexed = true;
          if (local.operation_name) delete local.operation_name;
          updated++;
        }
      }

      await save(data);
      return json({ success: true, updated_count: updated, total_remote_documents: docs.length });
    }

    // DELETE DOCUMENT
    if (pathname.startsWith("/stores/") && pathname.includes("/documents/") && method === "DELETE") {
      const segments = pathname.split("/");
      const storeName = segments[2];
      const documentId = segments[4];
      const apiKey = url.searchParams.get("api_key") || env.GEMINI_API_KEY;
      if (!apiKey) return json({ error: "Missing api_key" }, 400);

      const data = await load();
      const store = data.file_stores[storeName];
      if (!store) return json({ error: "Store not found" }, 404);

      const fsStore = store.file_search_store_name;
      const deleteURL = `${GEMINI_REST_BASE}/${fsStore}/documents/${documentId}?force=true&key=${encodeURIComponent(apiKey)}`;

      try {
        const resp = await fetch(deleteURL, { method: "DELETE" });
        if (![200, 204].includes(resp.status)) {
          const t = await resp.text();
          return json({ success: false, error: t }, resp.status);
        }
      } catch (e) {
        return json({ success: false, error: e.toString() }, 500);
      }

      // remove locally
      store.files = store.files.filter((f) => f.document_id !== documentId);
      await save(data);
      return json({ success: true, deleted_document_id: documentId });
    }

    // DELETE STORE
    if (pathname.startsWith("/stores/") && method === "DELETE") {
      const segments = pathname.split("/");
      const storeName = segments[2];
      const apiKey = url.searchParams.get("api_key") || env.GEMINI_API_KEY;
      if (!apiKey) return json({ error: "Missing api_key" }, 400);

      const data = await load();
      const store = data.file_stores[storeName];
      if (!store) return json({ error: "Store not found" }, 404);

      // attempt remote delete
      try {
        const delUrl = `${GEMINI_REST_BASE}/${store.file_search_store_name}?key=${encodeURIComponent(apiKey)}`;
        await fetch(delUrl, { method: "DELETE" });
      } catch (e) {
        // swallow
      }

      delete data.file_stores[storeName];
      if (data.current_store_name === storeName) data.current_store_name = null;
      await save(data);
      return json({ success: true, deleted_store: storeName });
    }

    // ASK (RAG) via models.generateContent endpoint (REST)
    if (pathname === "/ask" && method === "POST") {
      try {
        const body = await request.json();
        const apiKey = body.api_key || env.GEMINI_API_KEY;
        const stores = body.stores || [];
        const question = body.question;
        const system_prompt = body.system_prompt;

        if (!apiKey) return json({ error: "Missing api_key" }, 400);
        if (!question) return json({ error: "Missing question" }, 400);

        const data = await load();
        const fsStoreNames = [];
        for (const s of stores) {
          if (data.file_stores?.[s]?.file_search_store_name) {
            fsStoreNames.push(data.file_stores[s].file_search_store_name);
          }
        }
        if (fsStoreNames.length === 0) {
          return json({ error: "No valid File Search stores found for provided store names." }, 400);
        }

        const genUrl = `${GEMINI_REST_BASE}/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(apiKey)}`;
        const payload = {
          contents: [{ parts: [{ text: question }] }],
          tools: [
            {
              file_search: {
                file_search_store_names: fsStoreNames,
              },
            },
          ],
        };
        // include system instruction if provided
        if (system_prompt) {
          payload["config"] = { systemInstruction: system_prompt };
        }

        const resp = await fetch(genUrl, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!resp.ok) {
          const t = await resp.text();
          return json({ error: `generateContent failed: ${t}` }, resp.status);
        }
        const j = await resp.json();

        // Extract text and grounding metadata if present
        // generateContent responses vary; try to read likely fields
        const text = j?.candidates?.[0]?.output?.[0]?.content || j?.output?.[0]?.content || j?.text || "";
        const grounding =
          j?.candidates?.[0]?.grounding_metadata || j?.candidates?.[0]?.groundingMetadata || null;

        return json({ success: true, response_text: text, grounding_metadata: grounding });
      } catch (e) {
        return json({ error: e.toString() }, 500);
      }
    }

    // Default: not found
    return json({ error: "Route not found" }, 404);
  },
};
