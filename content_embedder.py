"""
Content transcript embedder — Modal GPU app.

Turns Amazon video transcripts (WebVTT URLs on catalog.contents.transcript_url)
into BGE-M3 dense vectors and upserts them as documents into the Pinecone
`content` index (document schema: dense 1024/cosine + BM25 title/transcript).

Namespace contract (docs/content_vectors.md in the scripts repo):
- every document is ALWAYS upserted to the `all` namespace first
- documents flagged is_logie are then upserted VERBATIM to `logie-creators`
- every processed batch is archived as parquet (id, vector, text, metadata)
  to R2 — the rebuild-insurance artifact, since the index schema is immutable

Two entry points share one pipeline:
- POST web endpoint `embed_batch` (shared-secret header) — hourly dispatcher
  batches from the scripts server (pinecone/content_embed_dispatch.py)
- `backfill_shard` — reads a JSONL shard from R2, used by the one-time
  backfill fan-out:  modal run content_embedder.py::backfill --prefix ...

Modal secrets required:
    modal secret create pinecone-content PINECONE_API_KEY=... PINECONE_CONTENT_HOST=...
    modal secret create content-embedder-auth CONTENT_EMBED_AUTH_KEY=...
    (object-storage-credentials already exists — R2 OBJECT_STORAGE_* keys)
"""
import json
import os
import time
import uuid

import modal

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 64
TRANSCRIPT_MAX_CHARS = int(os.environ.get("PINECONE_CONTENT_TRANSCRIPT_MAX_CHARS", "6000"))
# Per-container fetch politeness. During a 20–30 container backfill fan-out the
# aggregate is ~FETCH_CONCURRENCY × N_containers; kept modest because we hit
# Amazon's public media CDN, not our own infra. Override via the pinecone-content
# secret if we ever see throttling.
FETCH_CONCURRENCY = int(os.environ.get("CONTENT_FETCH_CONCURRENCY", "12"))
# Paid-proxy (Oxylabs) second pass runs gentler — it only handles the throttled
# remainder and we don't want to burn the 8GB/mo cap or hammer the unblocker.
FALLBACK_CONCURRENCY = int(os.environ.get("CONTENT_FALLBACK_CONCURRENCY", "6"))
FETCH_TIMEOUT_SECS = 20
FETCH_RETRIES = 3
FETCH_USER_AGENT = os.environ.get(
    "CONTENT_FETCH_USER_AGENT",
    "Mozilla/5.0 (compatible; LogieContentIndexer/1.0; +https://logie.ai)",
)
UPSERT_MAX_DOCS = 100
UPSERT_MAX_BYTES = 1_500_000  # documents API hard limit is 2 MB/request
PINECONE_SCHEMA_API_VERSION = "2026-01.alpha"
NAMESPACE_ALL = "all"
NAMESPACE_LOGIE = "logie-creators"
ARCHIVE_PREFIX = "content-vectors/parquet"
MAX_ITEMS_PER_REQUEST = 256

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # torch >= 2.6 required: transformers refuses torch.load below that
        # (CVE-2025-32434) and bge-m3 ships a .bin checkpoint path.
        "torch==2.7.1",
        "sentence-transformers==5.0.0",
        "safetensors",
        "httpx==0.27.0",
        "webvtt-py==0.4.6",
        "boto3==1.34.0",
        "pyarrow==16.1.0",
        "fastapi[standard]",
    )
)

app = modal.App("content-embedder", image=image)
model_cache = modal.Volume.from_name("bge-m3-model", create_if_missing=True)

SECRETS = [
    modal.Secret.from_name("pinecone-content"),
    modal.Secret.from_name("content-embedder-auth"),
    # Dedicated R2 secret sourced from the repo .env (OBJECT_STORAGE_*), kept
    # separate from the shared object-storage-credentials the video-extractor
    # app uses so the two can't drift into each other.
    modal.Secret.from_name("content-r2"),
]


# ---------------------------------------------------------------------------
# Transcript fetch + parse (no GPU needed, plain helpers)
# ---------------------------------------------------------------------------

def _parse_vtt(vtt_text):
    """WebVTT -> plain text. Strips cue timing, collapses rolling-caption
    repeats and normalizes whitespace. Returns '' for cue-less files."""
    import io

    import webvtt

    lines = []
    try:
        for caption in webvtt.read_buffer(io.StringIO(vtt_text)):
            for line in caption.text.splitlines():
                line = " ".join(line.split())
                if line and (not lines or lines[-1] != line):
                    lines.append(line)
    except Exception:
        return ""
    return " ".join(lines).strip()


async def _fetch_all(items, proxy_url=None, proxy_mode="standard", concurrency=None):
    """Fetch + parse every item's transcript_url concurrently.
    Mutates each item: adds `text`, `status` ('ok'|'empty'|'fetch_failed') and
    `_bytes`. Returns total bytes downloaded (for bandwidth accounting).

    Direct fetches from a single Modal container IP get soft-throttled by
    Amazon's CDN with 200-empty bodies at volume; a rotating proxy exits from a
    fresh IP per request and returns the real transcript.
      proxy_mode='standard'   -> Oxylabs-style CONNECT proxy: keep https, verify on
      proxy_mode='scrapingdog'-> ScrapingDog proxy: http:// target + verify off
    """
    import asyncio
    import random

    import httpx

    sem = asyncio.Semaphore(concurrency or FETCH_CONCURRENCY)
    use_proxy = bool(proxy_url)
    flip_http = use_proxy and proxy_mode == "scrapingdog"
    total_bytes = 0

    def target(u):
        if flip_http and u.startswith("https://"):
            return "http://" + u[len("https://"):]
        return u

    async def fetch_one(client, item):
        nonlocal total_bytes
        url = (item.get("transcript_url") or "").strip()
        if not url:
            item["status"], item["text"] = "empty", ""
            return
        async with sem:
            for attempt in range(FETCH_RETRIES + 1):
                try:
                    resp = await client.get(target(url), timeout=FETCH_TIMEOUT_SECS)
                    if resp.status_code == 200:
                        total_bytes += len(resp.content)
                        text = _parse_vtt(resp.text)
                        if text:
                            item["status"], item["text"] = "ok", text
                        else:
                            item["status"], item["text"] = "empty", ""
                        return
                    if resp.status_code in (403, 404, 410):
                        # permanent: dead/removed asset — don't hammer it
                        item["status"], item["text"] = "fetch_failed", ""
                        item["error"] = f"http {resp.status_code}"
                        return
                    if resp.status_code in (429, 503):
                        # throttled — honor Retry-After when present, else backoff
                        item["error"] = f"http {resp.status_code}"
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            await asyncio.sleep(min(int(retry_after), 30))
                            continue
                    else:
                        item["error"] = f"http {resp.status_code}"
                except Exception as exc:  # timeout / network — retry
                    item["error"] = str(exc)[:200]
                if attempt < FETCH_RETRIES:
                    # exponential backoff with jitter to avoid synchronized retries
                    await asyncio.sleep(min(2 ** attempt, 8) + random.random())
            item["status"] = "fetch_failed"
            item["text"] = ""

    headers = {"User-Agent": FETCH_USER_AGENT, "Accept": "text/vtt,text/plain,*/*"}
    client_kwargs = dict(follow_redirects=True, headers=headers)
    if use_proxy:
        client_kwargs["proxy"] = proxy_url
        # unblocker/scrapingdog MITM TLS (their cert won't match the target),
        # so skip verification; a plain residential CONNECT proxy keeps it on.
        client_kwargs["verify"] = proxy_mode == "standard"
        # Force a fresh connection per request so a rotating residential proxy
        # hands out a new exit IP each time (keep-alive would pin one IP).
        client_kwargs["limits"] = httpx.Limits(max_keepalive_connections=0)
    async with httpx.AsyncClient(**client_kwargs) as client:
        await asyncio.gather(*(fetch_one(client, item) for item in items))
    return total_bytes


# ---------------------------------------------------------------------------
# Pinecone documents API
# ---------------------------------------------------------------------------

def _pinecone_headers():
    return {
        "Api-Key": os.environ["PINECONE_API_KEY"],
        "Content-Type": "application/json",
        "X-Pinecone-Api-Version": PINECONE_SCHEMA_API_VERSION,
    }


def _build_document(item, vector, embedded_at_ts):
    """One Pinecone document. Undeclared fields become filterable metadata;
    None values are dropped (the documents API rejects nulls)."""
    doc = {
        "_id": item["content_id"],
        "dense": vector,
        "title": (item.get("title") or "")[:2000],
        "transcript": item["text"][:TRANSCRIPT_MAX_CHARS],
        "is_logie": bool(item.get("is_logie")),
        "embedded_at_ts": embedded_at_ts,
        "model": "bge-m3",
    }
    for key, value in (item.get("metadata") or {}).items():
        if value is not None and key not in doc:
            doc[key] = value
    return doc


def _upsert_documents(host, namespace, docs, dry_run=False):
    """Chunk by doc count AND serialized size (2 MB request cap), with retries."""
    import httpx

    url = f"https://{host}/namespaces/{namespace}/documents/upsert"
    batch, batch_bytes = [], 0
    batches = []
    for doc in docs:
        doc_bytes = len(json.dumps(doc))
        if batch and (len(batch) >= UPSERT_MAX_DOCS or batch_bytes + doc_bytes > UPSERT_MAX_BYTES):
            batches.append(batch)
            batch, batch_bytes = [], 0
        batch.append(doc)
        batch_bytes += doc_bytes
    if batch:
        batches.append(batch)

    if dry_run:
        return {"batches": len(batches), "docs": len(docs), "dry_run": True}

    with httpx.Client(timeout=120) as client:
        for chunk in batches:
            for attempt in range(4):
                resp = client.post(url, headers=_pinecone_headers(), json={"documents": chunk})
                if resp.status_code < 300:
                    break
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"pinecone upsert {namespace} failed ({resp.status_code}): {resp.text[:300]}"
                )
    return {"batches": len(batches), "docs": len(docs)}


# ---------------------------------------------------------------------------
# R2 parquet archive
# ---------------------------------------------------------------------------

def _get_r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_ACCESS_KEY"],
    )


def _archive_parquet(key, ok_items, vectors, embedded_at_ts):
    """id, vector, title, transcript, metadata(json) -> R2. Rebuild insurance."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "id": [item["content_id"] for item in ok_items],
            "values": pa.array(vectors, type=pa.list_(pa.float32())),
            "title": [(item.get("title") or "") for item in ok_items],
            "transcript": [item["text"][:TRANSCRIPT_MAX_CHARS] for item in ok_items],
            "product_anchor": [(item.get("product_anchor") or "") for item in ok_items],
            "is_logie": [bool(item.get("is_logie")) for item in ok_items],
            "metadata": [json.dumps(item.get("metadata") or {}) for item in ok_items],
            "embedded_at_ts": [embedded_at_ts] * len(ok_items),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    buf.seek(0)
    bucket = os.environ.get("OBJECT_STORAGE_BUCKET", "logie-users")
    _get_r2_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L4",
    timeout=3600,
    secrets=SECRETS,
    volumes={"/root/.cache/huggingface": model_cache},
    scaledown_window=120,
)
class Embedder:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME, device="cuda")
        self.model.half()
        model_cache.commit()

    def _embed(self, texts):
        vectors = self.model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # round to shrink JSON payloads; harmless for cosine at fp16 precision
        return [[round(float(x), 6) for x in vec] for vec in vectors]

    def _process(self, items, archive_key, dry_run=False, ns_all=NAMESPACE_ALL, ns_logie=NAMESPACE_LOGIE,
                 debug=False, proxy_url=None, proxy_mode="standard",
                 fallback_proxy_url=None, fallback_mode="standard"):
        """The single shared pipeline: fetch -> parse -> embed -> upsert -> archive.

        Two-pass fetch to minimize paid proxy bandwidth: pass 1 uses proxy_url
        (usually None = Modal direct, free); anything that comes back empty or
        failed — i.e. what Amazon's CDN throttled — is re-fetched in pass 2
        through fallback_proxy_url (Oxylabs, paid, rotating). Only the throttled
        remainder ever costs bandwidth."""
        import asyncio

        started = time.time()
        embedded_at_ts = int(started)
        host = os.environ["PINECONE_CONTENT_HOST"]

        def compose(item):
            """Dense embed text — anchor-LAST composition (locked 2026-07-13,
            A/B/C eval: transcript keeps the frame so same-product videos stay
            distinguishable; the product anchor still adds product/category
            mass so product-anonymous transcripts become findable). Transcript
            is pre-cropped BEFORE composing so the anchor always survives —
            never crop the composed text."""
            parts = [(item.get("title") or "").strip()]
            parts.append(f"Transcript: {item['text'][:TRANSCRIPT_MAX_CHARS]}")
            anchor = (item.get("product_anchor") or "").strip()
            if anchor:
                parts.append(anchor)
            return "\n".join(p for p in parts if p)

        direct_bytes = asyncio.run(_fetch_all(items, proxy_url=proxy_url, proxy_mode=proxy_mode))
        fallback_bytes = 0
        fallback_used = 0
        if fallback_proxy_url:
            retry = [it for it in items if it["status"] in ("empty", "fetch_failed")]
            if retry:
                fallback_used = len(retry)
                for it in retry:
                    it.pop("status", None)
                    it.pop("text", None)
                fallback_bytes = asyncio.run(
                    _fetch_all(retry, proxy_url=fallback_proxy_url, proxy_mode=fallback_mode,
                               concurrency=FALLBACK_CONCURRENCY)
                )
        ok_items = [item for item in items if item["status"] == "ok"]

        if ok_items:
            texts = [compose(item) for item in ok_items]
            vectors = self._embed(texts)
            docs = [
                _build_document(item, vec, embedded_at_ts)
                for item, vec in zip(ok_items, vectors)
            ]
            if debug:
                self._debug_doc = {
                    "keys": sorted(docs[0].keys()),
                    "dense_head": docs[0]["dense"][:5],
                    "dense_len": len(docs[0]["dense"]),
                    "dense_types": [type(x).__name__ for x in docs[0]["dense"][:3]],
                    "title": docs[0].get("title"),
                    "transcript_head": (docs[0].get("transcript") or "")[:150],
                    "serialized_bytes": len(json.dumps(docs[0])),
                }
            # Contract: `all` first, always; then the verbatim logie copy.
            _upsert_documents(host, ns_all, docs, dry_run=dry_run)
            logie_docs = [d for d, item in zip(docs, ok_items) if item.get("is_logie")]
            if logie_docs:
                _upsert_documents(host, ns_logie, logie_docs, dry_run=dry_run)
            if not dry_run and archive_key:
                _archive_parquet(archive_key, ok_items, vectors, embedded_at_ts)
            for item in ok_items:
                item["status"] = "embedded"

        results = [
            {
                "content_id": item["content_id"],
                "status": item["status"],
                "error": item.get("error"),
                "text_chars": len(item.get("text") or ""),
            }
            for item in items
        ]
        counts = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        outcome = {
            "results": results,
            "counts": counts,
            "logie_docs": sum(1 for item in ok_items if item.get("is_logie")),
            "duration_secs": round(time.time() - started, 2),
            "dry_run": dry_run,
            "direct_bytes": direct_bytes,
            "fallback_used": fallback_used,      # items re-fetched via paid proxy
            "fallback_bytes": fallback_bytes,    # paid proxy bandwidth this shard
        }
        if debug:
            if getattr(self, "_debug_doc", None):
                outcome["debug_doc"] = self._debug_doc
            outcome["debug_fetch"] = [
                {"content_id": item["content_id"], **(item.get("_fetch_debug") or {})}
                for item in items
            ]
        return outcome

    @modal.fastapi_endpoint(method="POST")
    def embed_batch(self, payload: dict):
        """Hourly entry point. Body: {"items": [...], "auth_key": str, "dry_run": bool}.
        Auth: payload["auth_key"] must match CONTENT_EMBED_AUTH_KEY."""
        from fastapi import HTTPException

        if payload.get("auth_key") != os.environ.get("CONTENT_EMBED_AUTH_KEY"):
            raise HTTPException(status_code=401, detail="unauthorized")
        items = payload.get("items") or []
        if not items:
            return {"results": [], "counts": {}, "duration_secs": 0}
        if len(items) > MAX_ITEMS_PER_REQUEST:
            raise HTTPException(status_code=400, detail=f"max {MAX_ITEMS_PER_REQUEST} items per request")
        # Optional namespace override for staging validation runs; production
        # dispatch omits these and gets the `all` + `logie-creators` contract.
        ns_all = payload.get("namespace_all") or NAMESPACE_ALL
        ns_logie = payload.get("namespace_logie") or NAMESPACE_LOGIE
        archive = payload.get("archive", True)
        archive_key = (
            f"{ARCHIVE_PREFIX}/incremental/"
            f"{time.strftime('%Y%m%d')}/{uuid.uuid4().hex}.parquet"
        ) if archive else None
        return self._process(
            items,
            archive_key,
            dry_run=bool(payload.get("dry_run")),
            ns_all=ns_all,
            ns_logie=ns_logie,
            debug=bool(payload.get("debug")),
            proxy_url=payload.get("proxy_url") or None,
            proxy_mode=payload.get("proxy_mode") or "standard",
            fallback_proxy_url=payload.get("fallback_proxy_url") or None,
            fallback_mode=payload.get("fallback_mode") or "unblocker",
        )

    @modal.fastapi_endpoint(method="POST")
    def embed_query(self, payload: dict):
        """Embed a search query string -> 1024-dim BGE-M3 vector.
        Body: {"query": str, "auth_key": str}. Used by the scripts server /
        dashboard search path so the 2.3GB model never loads on the app box."""
        from fastapi import HTTPException

        if payload.get("auth_key") != os.environ.get("CONTENT_EMBED_AUTH_KEY"):
            raise HTTPException(status_code=401, detail="unauthorized")
        query = (payload.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query required")
        vector = self._embed([query])[0]
        return {"vector": vector, "model": "bge-m3", "dimension": len(vector)}

    @modal.method()
    def backfill_shard(self, shard_key: str, fallback_proxy_url: str = None,
                       fallback_mode: str = "standard"):
        """Process one JSONL shard from R2 (one item per line). Idempotent:
        skips if the results file already exists (safe fan-out resume).

        Pass 1 fetches direct from Modal's IP (free). Whatever Amazon throttles
        (200-empty) is re-fetched through fallback_proxy_url (Oxylabs, paid) so
        we only spend proxy bandwidth on the blocked remainder."""
        bucket = os.environ.get("OBJECT_STORAGE_BUCKET", "logie-users")
        results_key = shard_key.replace("/shards/", "/results/") + ".results.json"
        r2 = _get_r2_client()
        try:
            r2.head_object(Bucket=bucket, Key=results_key)
            return {"shard": shard_key, "skipped": True}
        except Exception:
            pass

        body = r2.get_object(Bucket=bucket, Key=shard_key)["Body"].read().decode("utf-8")
        items = [json.loads(line) for line in body.splitlines() if line.strip()]
        shard_name = shard_key.rsplit("/", 1)[-1].replace(".jsonl", "")
        archive_key = f"{ARCHIVE_PREFIX}/backfill/{shard_name}.parquet"
        outcome = self._process(
            items, archive_key,
            fallback_proxy_url=fallback_proxy_url or None,
            fallback_mode=fallback_mode,
        )
        outcome["shard"] = shard_key
        r2.put_object(Bucket=bucket, Key=results_key, Body=json.dumps(outcome).encode("utf-8"))
        return outcome


# ---------------------------------------------------------------------------
# Backfill driver:  modal run content_embedder.py::backfill --prefix content-vectors/backfill/shards/
# ---------------------------------------------------------------------------

@app.function(secrets=SECRETS, timeout=6 * 3600)
def list_shards(prefix: str):
    bucket = os.environ.get("OBJECT_STORAGE_BUCKET", "logie-users")
    r2 = _get_r2_client()
    keys, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        page = r2.list_objects_v2(**kwargs)
        keys.extend(o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".jsonl"))
        if not page.get("IsTruncated"):
            return sorted(keys)
        token = page.get("NextContinuationToken")


@app.local_entrypoint()
def backfill(prefix: str = "content-vectors/backfill/shards/",
             fallback_proxy_url: str = "", fallback_mode: str = "standard"):
    # Two-pass: Modal fetches direct (free); the throttled remainder is re-fetched
    # through the Oxylabs fallback. The fallback URL is resolved client-side and
    # passed per shard so creds never live in a Modal secret or in R2. Set
    # CONTENT_FALLBACK_PROXY_URL in the shell (built from OXYLABS_* in .env)
    # before `modal run`.
    fallback_proxy_url = fallback_proxy_url or os.environ.get("CONTENT_FALLBACK_PROXY_URL", "")
    shard_keys = list_shards.remote(prefix)
    print(f"{len(shard_keys)} shards under {prefix} | fallback={'Oxylabs' if fallback_proxy_url else 'OFF'}")
    embedder = Embedder()
    done = failed = 0
    totals = {}
    fb_bytes = fb_used = 0
    for outcome in embedder.backfill_shard.map(
        shard_keys,
        kwargs={"fallback_proxy_url": fallback_proxy_url, "fallback_mode": fallback_mode},
        return_exceptions=True,
    ):
        if isinstance(outcome, Exception):
            failed += 1
            print(f"SHARD FAILED: {outcome}")
            continue
        done += 1
        if not outcome.get("skipped"):
            for k, v in (outcome.get("counts") or {}).items():
                totals[k] = totals.get(k, 0) + v
            fb_bytes += outcome.get("fallback_bytes") or 0
            fb_used += outcome.get("fallback_used") or 0
        tag = "skipped" if outcome.get("skipped") else outcome.get("counts")
        print(f"[{done}/{len(shard_keys)}] {outcome['shard']}: {tag}")
    print(f"done={done} failed={failed} totals={totals}")
    print(f"Oxylabs fallback: {fb_used} items, {fb_bytes/1e6:.1f} MB  "
          f"(8GB/mo cap → this run used {fb_bytes/8e9*100:.2f}% of it)")
