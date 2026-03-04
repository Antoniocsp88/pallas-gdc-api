from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote_plus

import requests
from fastapi import Body, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

# -----------------------------------------------------------------------------
# App & Config
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Pallas GDC API",
    version="1.0.0",
    description="Thin proxy over NCI GDC API with download helpers",
)

# (Optional) CORS for local UI/dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GDC_BASE = "https://api.gdc.cancer.gov"
GDC_STATUS = f"{GDC_BASE}/status"
GDC_PROJECTS = f"{GDC_BASE}/projects"
GDC_CASES = f"{GDC_BASE}/cases"
GDC_FILES = f"{GDC_BASE}/files"
GDC_ANNOTATIONS = f"{GDC_BASE}/annotations"
GDC_DATA = f"{GDC_BASE}/data"
GDC_MANIFEST = f"{GDC_BASE}/manifest"

SESSION = requests.Session()
TIMEOUT = 60  # seconds

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _bad_request(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)


def _parse_filters(filters: Optional[str]) -> Optional[str]:
    """
    Accepts a raw JSON string or URL-encoded JSON.
    Returns a JSON string (round-tripped to ensure validity) or None.
    """
    if not filters:
        return None
    try:
        decoded = unquote_plus(filters)
        obj = json.loads(decoded)
        return json.dumps(obj)
    except Exception:
        raise _bad_request(f"Value for 'filters' is not valid JSON: {filters}")


def _gdc_get(path: str, params: Optional[Dict[str, Any]] = None) -> JSONResponse:
    """
    Wrapper for GETs against GDC that returns JSONResponse or raises 502.
    path may be absolute (starts with 'http') or relative ('/projects').
    """
    url = path if path.startswith("http") else f"{GDC_BASE}{path}"
    try:
        r = SESSION.get(url, params=params or {}, timeout=TIMEOUT)
        r.raise_for_status()
        return JSONResponse(content=r.json())
    except requests.HTTPError as he:
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            raise HTTPException(status_code=502, detail=f"GDC error {r.status_code}: {he}")  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


def _filename_from_disposition(dispo: Optional[str], fallback: str) -> str:
    """
    Extract filename="..." from a Content-Disposition header if present.
    """
    if not dispo:
        return fallback
    try:
        parts = [p.strip() for p in dispo.split(";")]
        for p in parts:
            if p.lower().startswith("filename="):
                val = p.split("=", 1)[1].strip().strip('"')
                return val or fallback
    except Exception:
        pass
    return fallback


def _stream_gdc_file(file_id: str) -> Tuple[requests.Response, str]:
    """
    Perform a streaming GET to /data/{file_id}, returning the response and filename.
    """
    url = f"{GDC_DATA}/{file_id}"
    r = SESSION.get(url, stream=True, timeout=TIMEOUT)
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail=f"{file_id} not found")
    try:
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GDC data error ({file_id}): {e}")
    fname = _filename_from_disposition(r.headers.get("Content-Disposition"), f"{file_id}.dat")
    return r, fname


def _download_into_tar(ids: List[str]) -> bytes:
    """
    Download each file id from GDC /data/{id} and pack into a .tar bundle.
    Returns the tar bytes.
    """
    buf = io.BytesIO()
    with tarfile.open(mode="w", fileobj=buf) as tf:
        for fid in ids:
            resp, fname = _stream_gdc_file(fid)
            file_bytes = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_bytes.write(chunk)
            size = file_bytes.tell()
            file_bytes.seek(0)

            info = tarfile.TarInfo(name=fname)
            info.size = size
            info.mtime = int(datetime.utcnow().timestamp())
            tf.addfile(info, fileobj=file_bytes)
    buf.seek(0)
    return buf.read()


def _ids_from_query(ids: Optional[List[str]] = None, ids_csv: Optional[str] = None) -> List[str]:
    """
    Accept ids as repeated query params (?ids=a&ids=b) OR a single CSV string (?ids_csv=a,b).
    Also tolerates a single ?ids=comma,separated if only one element was provided.
    """
    collected: List[str] = []
    if ids:
        if len(ids) == 1 and ("," in ids[0]):
            collected.extend([x.strip() for x in ids[0].split(",") if x.strip()])
        else:
            collected.extend([x.strip() for x in ids if x.strip()])
    if ids_csv:
        collected.extend([x.strip() for x in ids_csv.split(",") if x.strip()])

    seen = set()
    uniq: List[str] = []
    for x in collected:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _stream_manifest_from_gdc(ids: List[str]) -> StreamingResponse:
    """
    FIX: Generate a manifest by POSTing to upstream GDC /manifest with JSON body {"ids":[...]}.
    This avoids the upstream 405 caused by GET /manifest?ids=...
    """
    if not ids:
        raise _bad_request("Provide at least one file id.")

    try:
        r = SESSION.post(
            GDC_MANIFEST,
            json={"ids": ids},
            stream=True,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        # If upstream returns an error, surface it cleanly.
        if r.status_code >= 400:
            # Try to parse JSON error; otherwise return text.
            try:
                err = r.json()
                raise HTTPException(status_code=r.status_code, detail=err)
            except Exception:
                raise HTTPException(status_code=r.status_code, detail=r.text[:2000] or f"GDC error {r.status_code}")

        # Preserve upstream headers when possible
        dispo = r.headers.get("Content-Disposition") or r.headers.get("content-disposition")
        ctype = r.headers.get("Content-Type") or r.headers.get("content-type") or "text/plain; charset=utf-8"

        # If no filename provided, set a reasonable default
        if not dispo:
            dispo = 'attachment; filename="gdc_manifest.txt"'

        headers = {"Content-Disposition": dispo}

        return StreamingResponse(r.iter_content(chunk_size=1024 * 64), media_type=ctype, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


# -----------------------------------------------------------------------------
# OpenAPI: remove the pre-rendered 422 block across all operations
# -----------------------------------------------------------------------------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    for path_item in schema.get("paths", {}).values():
        for op in path_item.values():
            if isinstance(op, dict) and "responses" in op:
                op["responses"].pop("422", None)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Pallas GDC API is running 🚀"}


@app.get("/status")
def gdc_status():
    try:
        r = SESSION.get(GDC_STATUS, timeout=TIMEOUT)
        r.raise_for_status()
        return JSONResponse(content=r.json())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


# --- Projects ---
@app.get("/projects/{project_id}")
def get_project(project_id: str = Path(..., description="GDC Project ID, e.g., TCGA-BRCA")):
    return _gdc_get(f"/projects/{project_id}")


@app.get("/projects", responses={200: {"description": "Projects search"}})
def list_projects(
    project_id: Optional[str] = Query(None, description="Simple filter: exact project_id (e.g., TCGA-BRCA)"),
    filters: Optional[str] = Query(
        None,
        description='JSON filters or URL-encoded JSON (e.g., {"op":"in","content":{"field":"project_id","value":["TCGA-BRCA"]}})',
    ),
    fields: Optional[str] = Query(None, description="Comma-separated fields to return"),
    size: int = Query(10, ge=1, le=1000),
    from_: int = Query(0, ge=0, alias="from"),
    sort: Optional[str] = Query(None),
):
    params: Dict[str, Any] = {"size": size, "from": from_}
    if fields:
        params["fields"] = fields
    if sort:
        params["sort"] = sort

    if filters:
        params["filters"] = _parse_filters(filters)
    elif project_id:
        params["filters"] = json.dumps({"op": "in", "content": {"field": "project_id", "value": [project_id]}})

    return _gdc_get("/projects", params=params)


# --- Cases / Files / Annotations (pass-through with optional filters) ---
@app.get("/cases")
def get_cases(
    filters: Optional[str] = Query(None, description="URL-encoded JSON filters"),
    fields: Optional[str] = Query(None, description="Comma-separated fields"),
    size: int = Query(10, ge=1, le=1000),
    from_: int = Query(0, ge=0, alias="from"),
    sort: Optional[str] = Query(None),
):
    params: Dict[str, Any] = {"size": size, "from": from_}
    if fields:
        params["fields"] = fields
    if sort:
        params["sort"] = sort
    if filters:
        params["filters"] = _parse_filters(filters)
    return _gdc_get("/cases", params=params)


@app.get("/files")
def get_files(
    filters: Optional[str] = Query(None, description="URL-encoded JSON filters"),
    fields: Optional[str] = Query(None, description="Comma-separated fields"),
    size: int = Query(10, ge=1, le=1000),
    from_: int = Query(0, ge=0, alias="from"),
    sort: Optional[str] = Query(None),
):
    params: Dict[str, Any] = {"size": size, "from": from_}
    if fields:
        params["fields"] = fields
    if sort:
        params["sort"] = sort
    if filters:
        params["filters"] = _parse_filters(filters)
    return _gdc_get("/files", params=params)


@app.get("/annotations")
def get_annotations(
    filters: Optional[str] = Query(None, description="URL-encoded JSON filters"),
    fields: Optional[str] = Query(None, description="Comma-separated fields"),
    size: int = Query(10, ge=1, le=1000),
    from_: int = Query(0, ge=0, alias="from"),
    sort: Optional[str] = Query(None),
):
    params: Dict[str, Any] = {"size": size, "from": from_}
    if fields:
        params["fields"] = fields
    if sort:
        params["sort"] = sort
    if filters:
        params["filters"] = _parse_filters(filters)
    return _gdc_get("/annotations", params=params)


# --- Manifest (FIXED) ---
@app.post(
    "/manifest",
    responses={200: {"description": "GDC manifest (plain text) download"}},
)
def post_manifest(
    payload: Dict[str, Any] = Body(..., description='JSON body like {"ids": ["id1","id2", ...]}'),
):
    ids = payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(x, str) and x.strip() for x in ids):
        raise _bad_request('Body must be: {"ids": ["id1","id2", ...]}')
    clean_ids = [x.strip() for x in ids]
    return _stream_manifest_from_gdc(clean_ids)


@app.get(
    "/manifest/{ids_csv}",
    responses={200: {"description": "GDC manifest (plain text) download"}},
)
def get_manifest(ids_csv: str = Path(..., description="One or more file_ids, comma-separated")):
    # Convenience GET: /manifest/<uuid> or /manifest/<uuid1,uuid2,...>
    ids = [x.strip() for x in ids_csv.split(",") if x.strip()]
    return _stream_manifest_from_gdc(ids)


# --- Slicing (produce helper URL; real slicing requires POST with regions to GDC) ---
@app.get("/slicing/{file_id}")
def slicing_url(file_id: str):
    return {"slice_url": f"{GDC_BASE}/slicing/view/{file_id}"}


# --- Data: single file download ---
@app.get("/data/{file_id}", responses={200: {"description": "File download"}})
def get_data_file(file_id: str = Path(..., description="GDC file_id to download")):
    r, fname = _stream_gdc_file(file_id)
    headers = {
        "Content-Disposition": f'attachment; filename="{fname}"',
        "Content-Type": r.headers.get("Content-Type", "application/octet-stream"),
    }
    return StreamingResponse(r.iter_content(chunk_size=1024 * 256), headers=headers)


# --- Data: multiple files via GET ---
@app.get("/data", responses={200: {"description": "Bundle download (.tar)"}})
def get_data_bundle(
    ids: Optional[List[str]] = Query(
        None, description="Repeat ?ids=... for multiple ids, or a single comma-separated string"
    ),
    ids_csv: Optional[str] = Query(None, description="Alternative: comma-separated ids in one parameter (ids_csv)"),
):
    id_list = _ids_from_query(ids, ids_csv)
    if not id_list:
        raise _bad_request("Provide at least one 'ids' value (repeatable) or 'ids_csv' CSV list.")
    tar_bytes = _download_into_tar(id_list)
    headers = {"Content-Disposition": 'attachment; filename="gdc_bundle.tar"'}
    return StreamingResponse(io.BytesIO(tar_bytes), media_type="application/x-tar", headers=headers)


# --- Data: multiple files via POST (JSON body) ---
@app.post("/data", responses={200: {"description": "Bundle download (.tar)"}})
def post_data_bundle(
    payload: Dict[str, Any] = Body(..., description='JSON body like {"ids": ["id1","id2", ...]}'),
):
    ids = payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(x, str) and x.strip() for x in ids):
        raise _bad_request('Body must be: {"ids": ["id1","id2", ...]}')
    tar_bytes = _download_into_tar([x.strip() for x in ids])
    headers = {"Content-Disposition": 'attachment; filename="gdc_bundle.tar"'}
    return StreamingResponse(io.BytesIO(tar_bytes), media_type="application/x-tar", headers=headers)


# -----------------------------------------------------------------------------
# Error normalization: convert RequestValidationError -> HTTP 400 (no 422s at runtime)
# -----------------------------------------------------------------------------
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    resp = await request_validation_exception_handler(request, exc)
    resp.status_code = 400
    return resp


# production add ons
if __name__ == "__main__":
    import os, uvicorn

    uvicorn.run(
        "pallas_gdc_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )