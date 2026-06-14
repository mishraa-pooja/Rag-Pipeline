# Financial Document Management API

A FastAPI service for **storing, managing, and semantically searching financial
documents** (reports, invoices, contracts) using a Retrieval-Augmented
Generation (RAG) pipeline.

- **API**: FastAPI + SQLAlchemy 2.0 (SQLite by default)
- **Auth**: JWT (HS256, algorithm-pinned) + bcrypt password hashing
- **RBAC**: 4 default roles (Admin / Financial Analyst / Auditor / Client),
  permission-keyed; everything denies-by-default
- **RAG**: LangChain text splitting → embeddings → Qdrant vector DB → cross-encoder rerank
  - Pluggable embed/rerank backend, switched by a single env var:
    - `local_bge` — in-process `BAAI/bge-small-en-v1.5` + `BAAI/bge-reranker-base` (default)
    - `cisco_aiverse` — Cisco's internal AIverse gateway over OAuth2: `google/embeddinggemma-300m` + `ibm-granite/granite-embedding-reranker-english-r2`
- **Document types**: PDF, DOCX, plain text

---

## 1. Quickstart (Docker Compose, recommended)

```bash
# 1. Configure environment (generate a real secret!)
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(64))"   # paste into JWT_SECRET_KEY
# also update BOOTSTRAP_ADMIN_PASSWORD to a strong value (12+ chars)

# 2. Start Qdrant + API
docker compose up -d --build

# 3. Open the interactive docs
open http://localhost:8000/docs
```

The API binds to `127.0.0.1:8000`, Qdrant to `127.0.0.1:6333`. On first start
the app creates the SQLite schema, seeds the 4 default roles + their permission
sets, and creates the bootstrap admin defined by `BOOTSTRAP_ADMIN_*`.

The first request that touches the RAG pipeline (index or search) will
download the embedding + reranker models — expect a one-time **~1.2 GB**
download into the container's HuggingFace cache. See §2.1 for an optional
pre-warm command that downloads them up front.

## 2. Quickstart (local Python)

**macOS / Linux**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit JWT_SECRET_KEY, BOOTSTRAP_ADMIN_PASSWORD

# Run Qdrant separately (docker is easiest):
docker run -d --name qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.12.4

uvicorn app.main:app --reload
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# edit JWT_SECRET_KEY, BOOTSTRAP_ADMIN_PASSWORD with notepad/VS Code

# Run Qdrant separately:
docker run -d --name qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.12.4

uvicorn app.main:app --reload
```

If PowerShell refuses to run `Activate.ps1`, allow scripts for the current user once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

### 2.1 Pre-downloading the BGE models (optional)

The embedding + reranker models download lazily on the first `/rag/*` call
(~1.2 GB total: ~130 MB for `bge-small-en-v1.5`, ~1.1 GB for
`bge-reranker-base`). To download them up front so the first user request
is fast — and to confirm `huggingface.co` is reachable from your machine
before you depend on it:

```bash
# Works the same on macOS, Linux, and Windows (any shell):
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base'); \
print('models cached')"
```

Models are cached under `%USERPROFILE%\.cache\huggingface\hub` on Windows and
`~/.cache/huggingface/hub` on macOS / Linux. Subsequent runs reuse the cache
(no network needed).

If this command fails with `CERTIFICATE_VERIFY_FAILED`, your network is
behind a TLS-inspection proxy. The app handles this at startup
(see `app/__init__.py`), but for the pre-warm one-liner you can either
(a) run it inside `docker compose run --rm api python -c "..."` so it picks
up the container's CA bundle, or (b) switch to
`EMBEDDING_PROVIDER=cisco_aiverse` once your OAuth client has FMSConsumer
access for `embeddinggemma-300m` + `granite-reranker`.

---

## 3. Architecture

```
                         ┌──────────────────────┐
        HTTP client ───► │   FastAPI (uvicorn)  │ ◄── JWT bearer (HS256)
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
       ┌──────▼──────┐      ┌───────▼───────┐     ┌───────▼───────┐
       │  Auth + RBAC│      │   Documents   │     │     RAG       │
       │  /auth/*    │      │  /documents/* │     │   /rag/*      │
       │  /roles/*   │      │  upload, CRUD │     │  index/search │
       │  /users/*   │      │  + metadata   │     │  + rerank     │
       └──────┬──────┘      └────────┬──────┘     └───┬───────────┘
              │                      │                │
              ▼                      ▼                ▼
       ┌──────────────┐     ┌────────────────┐  ┌────────────┐
       │  SQLite (SA) │     │ Local FS (UUID │  │   Qdrant   │
       │  users/roles │     │  filenames)    │  │  (chunks)  │
       │  documents   │     └────────────────┘  └────────────┘
       └──────────────┘
```

### Indexing pipeline

```
Document file (PDF / DOCX / TXT)
        │
        ▼  pypdf / python-docx / utf-8 decode
   Plain text  (≤ 1.5M chars)
        │
        ▼  LangChain RecursiveCharacterTextSplitter (chunk=800, overlap=120)
   Semantic chunks
        │
        ▼  embeddings provider (L2-normalized)
        │     local_bge      → BAAI/bge-small-en-v1.5  (384-d)
        │     cisco_aiverse  → google/embeddinggemma-300m  (768-d)
   Embeddings
        │
        ▼  Qdrant upsert (cosine, deterministic UUID5 point ids)
   Vector store
```

### Retrieval pipeline

```
   Query string
       │
       ▼  query-prefixed embedding (same provider as indexing)
   query vector
       │
       ▼  Qdrant top-K=20 (with metadata filters: company / type / doc_id)
   Candidate chunks
       │
       ▼  reranker provider (cross-encoder)
       │     local_bge      → BAAI/bge-reranker-base
       │     cisco_aiverse  → ibm-granite/granite-embedding-reranker-english-r2
   Top-5 reranked results
```

---

## 4. RBAC model

| Role               | Permissions                                                                                       |
|--------------------|---------------------------------------------------------------------------------------------------|
| `admin`            | everything (all 9 permission keys)                                                                |
| `financial_analyst`| `document:upload`, `document:read_any`, `document:edit`, `rag:index`, `rag:search`                |
| `auditor`          | `document:read_any`, `rag:search`                                                                 |
| `client`           | `document:read_own_company`, `rag:search` (scoped to own `company_name`)                          |

Permission keys are declared once in `app/core/permissions.py` and seeded into
the DB at startup. New users created via `POST /auth/register` get the
`client` role only — privilege escalation requires an admin to call
`POST /users/assign-role`.

---

## 5. API reference (end-to-end curl)

Below assumes the API is on `http://localhost:8000`. Replace `$TOKEN` with the
value returned by `/auth/login`.

### 5.1 Auth

```bash
# Register a regular (client) user
curl -sX POST http://localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{
        "email": "anjali@example.com",
        "username": "anjali",
        "password": "Sup3r-Strong!2026",
        "company_name": "Acme Corp"
      }'

# Login (works with either email or username as `identifier`)
TOKEN=$(curl -sX POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"identifier":"anjali","password":"Sup3r-Strong!2026"}' \
  | python -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')
echo $TOKEN
```

### 5.2 Roles & users (admin only)

```bash
# Get the admin token using the BOOTSTRAP_ADMIN_* credentials from .env
ADMIN_TOKEN=$(curl -sX POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"identifier":"admin","password":"<BOOTSTRAP_ADMIN_PASSWORD>"}' \
  | python -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')

# Create a custom role
curl -sX POST http://localhost:8000/roles/create \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"read_only","description":"Reader","permissions":["document:read_any","rag:search"]}'

# Promote anjali to financial_analyst
curl -sX POST http://localhost:8000/users/assign-role \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"<anjali_user_id>","role_name":"financial_analyst"}'

# Inspect a user's roles + permissions
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/users/<user_id>/roles
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/users/<user_id>/permissions
```

### 5.3 Documents

```bash
# Upload (analyst+ required)
curl -sX POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=Q4 2025 Earnings Report" \
  -F "company_name=Acme Corp" \
  -F "document_type=report" \
  -F "file=@./samples/q4_report.pdf"

# List
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/documents

# Get one
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/documents/<document_id>

# Metadata search
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/documents/search?company_name=Acme&document_type=report&limit=10"

# Delete (admin OR original uploader)
curl -sX DELETE -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/documents/<document_id>
```

### 5.4 RAG

```bash
# Index a document into Qdrant
curl -sX POST http://localhost:8000/rag/index-document \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"<document_id>"}'

# Semantic search with rerank (top 5 best chunks across the corpus)
curl -sX POST http://localhost:8000/rag/search \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"financial risk related to high debt ratio","top_k":5}'

# Fetch the full chunk list for one document (ordered)
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/rag/context/<document_id>

# Remove a document's embeddings from the vector store
curl -sX DELETE -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/rag/remove-document/<document_id>
```

---

## 6. Project layout

```
.
├── app/
│   ├── api/                  # routers
│   │   ├── auth.py           # /auth/*
│   │   ├── roles.py          # /roles/*  /users/*
│   │   ├── documents.py      # /documents/*
│   │   └── rag.py            # /rag/*
│   ├── core/
│   │   ├── security.py       # bcrypt + JWT (alg-pinned, iss/aud-checked)
│   │   ├── deps.py           # get_current_user, require_permission
│   │   └── permissions.py    # canonical permission keys + default roles
│   ├── models/               # SQLAlchemy 2.0 models
│   ├── schemas/              # Pydantic v2 request/response schemas
│   ├── services/
│   │   ├── file_storage.py   # safe upload (MIME + magic bytes + size + hash)
│   │   ├── extraction.py     # PDF / DOCX / TXT text extraction
│   │   ├── chunking.py       # LangChain semantic chunking
│   │   ├── embeddings.py     # BGE bi-encoder (lazy singleton)
│   │   ├── reranker.py       # BGE cross-encoder (lazy singleton)
│   │   ├── vector_store.py   # Qdrant wrapper (cosine, payload indexes)
│   │   └── rag_service.py    # orchestration: index + retrieve+rerank
│   ├── config.py             # pydantic-settings, secrets from env
│   ├── database.py           # engine + session
│   ├── seed.py               # idempotent role/permission/admin bootstrap
│   └── main.py               # FastAPI app factory + lifespan
├── data/                     # SQLite + uploaded files (gitignored)
├── docker-compose.yml        # api + qdrant
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 7. Security notes

Every control below maps to a real attack class and is enforced in code, not by
documentation alone.

| Concern                          | Control                                                                                              | Where                                  |
|----------------------------------|------------------------------------------------------------------------------------------------------|----------------------------------------|
| Hardcoded secrets                | All secrets read from env via `pydantic-settings`; `.env` is gitignored, `.env.example` is committed | `app/config.py`, `.gitignore`          |
| Weak password storage            | bcrypt via `passlib`; per-user salt; constant-time verify                                            | `app/core/security.py`                 |
| JWT algorithm confusion          | Algorithm **pinned** at decode; required claims `exp/iat/sub/iss/aud` enforced                       | `app/core/security.py`                 |
| Username enumeration             | Generic 401 on bad login; dummy bcrypt verify on unknown-user branch (timing equalization)           | `app/api/auth.py`                      |
| Self-privilege escalation        | Registration always assigns `client`; only `user:manage` can call `/users/assign-role`               | `app/api/auth.py`, `app/api/roles.py`  |
| Mass assignment                  | Pydantic `extra="forbid"` on all request models; ORM never accepts request bodies directly          | `app/schemas/*`                        |
| SQL injection                    | SQLAlchemy ORM with parameterized queries everywhere; no raw SQL with user input                     | all of `app/`                          |
| IDOR (document access)           | 404 (not 403) on cross-tenant access; client role scoped to own `company_name`                       | `app/api/documents.py`, `app/api/rag.py` |
| Unrestricted file upload         | MIME allow-list + magic-byte sniffing + size cap + server-generated UUID filename + sha256           | `app/services/file_storage.py`         |
| Path traversal                   | All file paths resolved against `upload_dir` and validated via `relative_to`                         | `app/services/file_storage.py`         |
| Files stored in web-root         | Uploads live in `./data/uploads`, not served by FastAPI; only retrievable via authenticated APIs     | `app/config.py` + no static mount      |
| Prompt-injection-style abuse     | RAG never executes retrieved text; results are plain JSON; queries are bounded in length             | `app/schemas/rag.py`, `app/api/rag.py` |
| DoS via huge documents           | `MAX_UPLOAD_BYTES` cap (default 20 MiB) + 1.5M-char extraction cap + bounded chunk size              | `app/services/file_storage.py`, `extraction.py` |
| Permissive CORS                  | Default `allow_origins=[]`; opt-in only                                                              | `app/main.py`                          |
| Container privilege              | Dockerfile drops to `appuser`; compose uses `no-new-privileges` + `cap_drop: ALL`; ports bound to 127.0.0.1 | `Dockerfile`, `docker-compose.yml` |

### Operational hardening to add before production

These are intentional follow-ups, not in scope for the assignment but called out
so reviewers see the gap:

1. **HTTPS + HSTS** in front of the API (reverse proxy: nginx / caddy / cloud LB).
2. **Rate limiting** on `/auth/*` and `/rag/search` (e.g. slowapi or gateway).
3. **Structured JSON logging** with correlation IDs and PII redaction (the
   current logger is the FastAPI default — easy to swap to `structlog`).
4. **Refresh tokens + revocation list** for long-lived sessions (the current
   model uses short-lived 30-min access tokens only).
5. **Alembic migrations** instead of `create_all`.
6. **Qdrant API key** turned on (`QDRANT__SERVICE__API_KEY` env var on the
   qdrant container, then set `QDRANT_API_KEY` in `.env`).

---

## 8. Config reference

| Env var                              | Default                                              | Purpose                                                                 |
|--------------------------------------|------------------------------------------------------|-------------------------------------------------------------------------|
| `JWT_SECRET_KEY`                     | _required_                                           | HMAC secret for JWTs. Min 32 chars. Generate fresh per env.            |
| `JWT_ALGORITHM`                      | `HS256`                                              | Pinned at decode time.                                                  |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`    | `30`                                                 | Access-token lifetime.                                                  |
| `JWT_ISSUER` / `JWT_AUDIENCE`        | `fin-doc-mgmt` / `fin-doc-mgmt-clients`              | Claim values, validated on decode.                                      |
| `BOOTSTRAP_ADMIN_EMAIL/USERNAME/PASSWORD` | example values; **must override password**       | Used only when the user table is empty.                                 |
| `DATABASE_URL`                       | `sqlite:///./data/app.db`                            | Any SQLAlchemy URL (e.g. PostgreSQL) works.                             |
| `UPLOAD_DIR`                         | `./data/uploads`                                     | File store. Never served by FastAPI.                                    |
| `MAX_UPLOAD_BYTES`                   | `20971520` (20 MiB)                                  | Per-file size cap.                                                      |
| `ALLOWED_UPLOAD_MIMES`               | `application/pdf,text/plain,docx`                    | MIME allow-list; everything else returns 415.                           |
| `QDRANT_URL` / `QDRANT_API_KEY`      | `http://localhost:6333` / _empty_                    | Vector store endpoint + optional API key.                               |
| `QDRANT_COLLECTION`                  | `financial_documents`                                | Collection name (auto-created at startup).                              |
| `EMBEDDING_PROVIDER`                 | `local_bge`                                          | `local_bge` or `cisco_aiverse`. Dispatches embed + rerank services.    |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM`  | `BAAI/bge-small-en-v1.5` / `384`                     | If you change the model, also change the dim and drop the collection.   |
| `RERANKER_MODEL`                     | `BAAI/bge-reranker-base`                             | Cross-encoder used for stage-2 ranking.                                 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`       | `800` / `120` (characters)                           | Chunking parameters.                                                    |
| `RETRIEVE_TOP_K` / `RERANK_TOP_K`    | `20` / `5`                                           | Stage-1 / stage-2 result counts.                                        |
| `CISCO_CLIENT_ID` / `_SECRET`        | _required if provider=cisco_aiverse_                 | Duo SSO OAuth2 client credentials.                                      |
| `CISCO_TOKEN_URL`                    | _required if provider=cisco_aiverse_                 | `https://sso-dbbfec7f.sso.duosecurity.com/oauth/<ID>/token`             |
| `CISCO_TOKEN_SCOPE`                  | `read write`                                         | **Must be `read write`** — `write` alone returns 401 from the gateway.  |
| `CISCO_EMBEDDING_BASE_URL` / `_MODEL`| `aiverse.cisco.com/embeddinggemma-300m/v1` / `google/embeddinggemma-300m` | OpenAI-compatible embeddings endpoint.             |
| `CISCO_RERANK_BASE_URL` / `_MODEL`   | `aiverse.cisco.com/granite-reranker/v1` / `ibm-granite/granite-embedding-reranker-english-r2` | Cohere-style rerank endpoint.   |
| `CISCO_REQUEST_TIMEOUT`              | `60`                                                 | Seconds.                                                                |
| `CISCO_RERANK_ENABLED`               | `true`                                               | Set `false` to skip reranking (useful while waiting on access).         |

---

## 9. RAG provider configuration

This service supports two interchangeable embed/rerank backends. The choice is
made at startup via `EMBEDDING_PROVIDER`; all other RAG code is unchanged.

### 9.1 `local_bge` (default, fully offline after first run)

In-process `sentence-transformers` with BGE models. Best for laptops or
disconnected environments.

```bash
# In .env
EMBEDDING_PROVIDER=local_bge
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
EMBEDDING_DIM=384
```

First indexing/search call downloads roughly **140 MB** from `huggingface.co`
into the HF cache (`~/.cache/huggingface` on the host or inside the container).
After that, no network is needed for the RAG stack.

### 9.2 `cisco_aiverse` (Cisco internal gateway)

Routes embed + rerank to Cisco's AIverse platform. No local model download.
Requires an OAuth2 client provisioned with **FMSConsumer access** for **both**
of the following models via the CPP Console:

- `embeddinggemma-300m`
- `granite-reranker`

```bash
# In .env
EMBEDDING_PROVIDER=cisco_aiverse
EMBEDDING_DIM=768                       # embeddinggemma-300m output dim

CISCO_CLIENT_ID=<duo-sso-client-id>
CISCO_CLIENT_SECRET=<duo-sso-client-secret>
CISCO_TOKEN_URL=https://sso-dbbfec7f.sso.duosecurity.com/oauth/<REDACTED>/token
CISCO_TOKEN_SCOPE=read write            # both scopes are required

CISCO_EMBEDDING_BASE_URL=https://aiverse.cisco.com/embeddinggemma-300m/v1
CISCO_EMBEDDING_MODEL=google/embeddinggemma-300m
CISCO_RERANK_BASE_URL=https://aiverse.cisco.com/granite-reranker/v1
CISCO_RERANK_MODEL=ibm-granite/granite-embedding-reranker-english-r2
```

Implementation lives in:

- `app/services/cisco_ai_client.py` — OAuth2 client_credentials with a
  thread-safe token cache (refreshes ~5 min before expiry).
- `app/services/embeddings.py` and `app/services/reranker.py` — dispatch on
  `EMBEDDING_PROVIDER`; both providers implement the same interface so the
  rest of the pipeline is unaware which backend is active.
- The Cisco reranker call uses the Cohere-style schema
  (`{model, query, documents}` → `{results: [{index, relevance_score}]}`)
  exactly as documented for `granite-reranker` `/v1/rerank`.

### 9.3 Switching providers on a live system

`EMBEDDING_PROVIDER` controls the vector dimension, so swapping providers
invalidates any indexed corpus. The Qdrant wrapper detects the mismatch on
startup and **automatically drops + recreates the collection** so the app
keeps working. This is safe for dev but destructive in production — for
production deployments, use a distinct `QDRANT_COLLECTION` name per provider
and migrate explicitly.

---

## 10. Troubleshooting

- **`JWT_SECRET_KEY must be at least 32 characters` on startup** — you didn't
  set it in `.env`. Generate one with
  `python -c "import secrets; print(secrets.token_urlsafe(64))"`.
- **First RAG call is slow (`local_bge`)** — the BGE embedding + reranker
  models are being downloaded into the HuggingFace cache. Subsequent calls
  are fast.
- **`huggingface.co` download fails with `CERTIFICATE_VERIFY_FAILED` or DNS
  block** — a corporate TLS-inspection proxy (Cisco Umbrella / Zscaler) is in
  the way. The app already propagates the OS CA bundle to
  `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` at startup (see `app/__init__.py`),
  which fixes the SSL case. If DNS itself is blocked, switch to
  `EMBEDDING_PROVIDER=cisco_aiverse` or pre-download the models on an
  unrestricted network and copy `~/.cache/huggingface` over.
- **`Connection refused` on Qdrant** — make sure `docker compose up -d qdrant`
  is running (or pass `QDRANT_URL` pointing at your instance).
- **Switching vector dimension** — if you change `EMBEDDING_MODEL`/provider,
  the Qdrant collection is auto-recreated on next startup. Use a distinct
  `QDRANT_COLLECTION` per provider for production.
- **Cisco AIverse returns `HTTP 401 "Invalid user key in JWT token"`** —
  there are **two** distinct causes that surface the exact same message:
    1. `CISCO_TOKEN_SCOPE` is `write` instead of `read write`. The gateway
       silently rejects tokens missing the `read` scope with this 401.
    2. Your OAuth client has no FMSConsumer access for the specific model.
       Per the Cisco wiki this should return 403, but in practice it returns
       this same 401. Request access via the CPP Console for both
       `embeddinggemma-300m` and `granite-reranker`. You can verify your
       credentials are otherwise valid by hitting a model you DO have access
       to (e.g. `llama-33-70b-instruct/v1/chat/completions`) with the same
       token — if that returns 200, the issue is per-model entitlement.
- **Cisco AIverse returns `HTTP 404`** — wrong URL. The path is always
  `https://aiverse.cisco.com/<model-slug>/v1/<endpoint>` (or
  `aiverse-nprd.cisco.com` for non-prod). The slug differs from the model
  name in the body (e.g. slug `embeddinggemma-300m`, body model
  `google/embeddinggemma-300m`).
