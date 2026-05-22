"""멀티유저/멀티세션 RAG 챗봇 — Supabase user 테이블 기반 로그인 및 세션 분리."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"
MODEL_NAME = "gpt-4o-mini"
VECTOR_BATCH_SIZE = 10
PBKDF2_ITERATIONS = 100_000

load_dotenv(dotenv_path=ENV_PATH)


def _get_secret(key: str) -> str:
    """Prefer Streamlit secrets, then environment variables."""
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value:
                return str(value).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()


def _apply_secrets_to_env() -> None:
    for key in ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"):
        val = _get_secret(key)
        if val:
            os.environ[key] = val


_apply_secrets_to_env()


def _writable_log_dir() -> Path | None:
    """Return a writable log directory (repo logs locally, /tmp on Streamlit Cloud)."""
    for candidate in (LOG_DIR, Path(tempfile.gettempdir()) / "multiusers_logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return None


def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    log_dir = _writable_log_dir()
    if log_dir is not None:
        log_name = f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = log_dir / log_name
        try:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.WARNING)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def check_env_keys() -> list[str]:
    missing: list[str] = []
    if not _get_secret("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not _get_secret("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not _get_secret("SUPABASE_ANON_KEY"):
        missing.append("SUPABASE_ANON_KEY")
    return missing


def get_supabase_client() -> Client | None:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_llm() -> ChatOpenAI:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model=MODEL_NAME, temperature=0.7, api_key=api_key)


def get_embeddings() -> OpenAIEmbeddings:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAIEmbeddings(api_key=api_key)


# ---------------------------------------------------------------------------
# Password hashing (no plaintext storage)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return secrets.compare_digest(digest.hex(), digest_hex)


# ---------------------------------------------------------------------------
# User auth (app `user` table — not Supabase Auth)
# ---------------------------------------------------------------------------
def register_user(client: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력해 주세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    existing = (
        client.table("user")
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False, "이미 사용 중인 아이디입니다."

    payload = {
        "login_id": login_id,
        "password_hash": hash_password(password),
    }
    resp = client.table("user").insert(payload).execute()
    if not resp.data:
        return False, "회원가입에 실패했습니다."
    return True, "회원가입이 완료되었습니다. 로그인해 주세요."


def login_user(client: Client, login_id: str, password: str) -> tuple[str | None, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return None, "아이디와 비밀번호를 입력해 주세요."

    resp = (
        client.table("user")
        .select("id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    row = rows[0]
    if not verify_password(password, row.get("password_hash", "")):
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    return str(row["id"]), "로그인되었습니다."


def get_logged_in_user_id() -> str | None:
    uid = st.session_state.get("logged_in_user_id")
    return str(uid) if uid else None


def logout_user() -> None:
    st.session_state.logged_in_user_id = None
    st.session_state.logged_in_login_id = None
    clear_screen_only()


def _session_belongs_to_user(client: Client, session_id: str, user_id: str) -> bool:
    resp = (
        client.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


# ---------------------------------------------------------------------------
# Supabase helpers (always scoped by user_id)
# ---------------------------------------------------------------------------
def fetch_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    resp = (
        client.table("chat_sessions")
        .select("id, title, file_names, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


def fetch_messages(client: Client, session_id: str, user_id: str) -> list[dict[str, str]]:
    if not _session_belongs_to_user(client, session_id, user_id):
        return []
    resp = (
        client.table("chat_messages")
        .select("role, content, sort_order")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("sort_order")
        .execute()
    )
    rows = resp.data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def persist_messages(
    client: Client,
    session_id: str,
    user_id: str,
    messages: list[dict[str, str]],
) -> None:
    if not _session_belongs_to_user(client, session_id, user_id):
        raise PermissionError("해당 세션에 접근할 수 없습니다.")
    client.table("chat_messages").delete().eq("session_id", session_id).eq(
        "user_id", user_id
    ).execute()
    if not messages:
        return
    payload = [
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": m["role"],
            "content": m["content"],
            "sort_order": idx,
        }
        for idx, m in enumerate(messages)
    ]
    client.table("chat_messages").insert(payload).execute()


def update_session_meta(
    client: Client,
    session_id: str,
    user_id: str,
    *,
    title: str | None = None,
    file_names: list[str] | None = None,
) -> None:
    if not _session_belongs_to_user(client, session_id, user_id):
        raise PermissionError("해당 세션에 접근할 수 없습니다.")
    patch: dict[str, Any] = {"updated_at": datetime.utcnow().isoformat()}
    if title is not None:
        patch["title"] = title
    if file_names is not None:
        patch["file_names"] = file_names
    client.table("chat_sessions").update(patch).eq("id", session_id).eq(
        "user_id", user_id
    ).execute()


def create_session_row(
    client: Client,
    user_id: str,
    title: str,
    file_names: list[str] | None = None,
) -> str:
    payload = {
        "user_id": user_id,
        "title": title,
        "file_names": file_names or [],
    }
    resp = client.table("chat_sessions").insert(payload).execute()
    if not resp.data:
        raise RuntimeError("세션 생성에 실패했습니다.")
    return str(resp.data[0]["id"])


def delete_session_row(client: Client, session_id: str, user_id: str) -> None:
    if not _session_belongs_to_user(client, session_id, user_id):
        raise PermissionError("해당 세션에 접근할 수 없습니다.")
    client.table("chat_sessions").delete().eq("id", session_id).eq(
        "user_id", user_id
    ).execute()


def list_vector_file_names(client: Client, session_id: str, user_id: str) -> list[str]:
    if not _session_belongs_to_user(client, session_id, user_id):
        return []
    resp = (
        client.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    return sorted({r["file_name"] for r in (resp.data or []) if r.get("file_name")})


def copy_vectors_to_session(
    client: Client,
    source_session_id: str,
    target_session_id: str,
    user_id: str,
) -> None:
    if not _session_belongs_to_user(client, source_session_id, user_id):
        return
    if not _session_belongs_to_user(client, target_session_id, user_id):
        raise PermissionError("대상 세션에 접근할 수 없습니다.")

    resp = (
        client.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(
            {
                "session_id": target_session_id,
                "file_name": row["file_name"],
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "embedding": row["embedding"],
            }
        )
        if len(batch) >= VECTOR_BATCH_SIZE:
            client.table("vector_documents").insert(batch).execute()
            batch = []
    if batch:
        client.table("vector_documents").insert(batch).execute()


def store_vectors_for_file(
    client: Client,
    session_id: str,
    user_id: str,
    file_name: str,
    splits: list[Document],
    embeddings: OpenAIEmbeddings,
) -> int:
    if not _session_belongs_to_user(client, session_id, user_id):
        raise PermissionError("해당 세션에 접근할 수 없습니다.")
    if not splits:
        return 0
    texts = [d.page_content for d in splits]
    vectors = embeddings.embed_documents(texts)
    stored = 0
    batch: list[dict[str, Any]] = []
    for doc, emb in zip(splits, vectors, strict=True):
        meta = dict(doc.metadata or {})
        batch.append(
            {
                "session_id": session_id,
                "file_name": file_name,
                "content": doc.page_content,
                "metadata": meta,
                "embedding": emb,
            }
        )
        if len(batch) >= VECTOR_BATCH_SIZE:
            client.table("vector_documents").insert(batch).execute()
            stored += len(batch)
            batch = []
    if batch:
        client.table("vector_documents").insert(batch).execute()
        stored += len(batch)
    return stored


def retrieve_documents(
    client: Client,
    session_id: str,
    user_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = 10,
) -> list[Document]:
    if not _session_belongs_to_user(client, session_id, user_id):
        return []
    query_emb = embeddings.embed_query(query)
    try:
        resp = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_emb,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        rows = resp.data or []
        return [
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "file_name": r.get("file_name", ""),
                    **(r.get("metadata") or {}),
                    "similarity": r.get("similarity"),
                },
            )
            for r in rows
            if r.get("content")
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC match_vector_documents failed: %s", exc)
        return _retrieve_documents_fallback(
            client, session_id, user_id, query, embeddings, k
        )


def _retrieve_documents_fallback(
    client: Client,
    session_id: str,
    user_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int,
) -> list[Document]:
    if not _session_belongs_to_user(client, session_id, user_id):
        return []
    resp = (
        client.table("vector_documents")
        .select("content, file_name, metadata, embedding")
        .eq("session_id", session_id)
        .limit(200)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return []
    query_emb = embeddings.embed_query(query)
    scored: list[tuple[float, Document]] = []
    for row in rows:
        emb = row.get("embedding")
        if not emb:
            continue
        sim = _cosine_similarity(query_emb, emb)
        scored.append(
            (
                sim,
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "file_name": row.get("file_name", ""),
                        **(row.get("metadata") or {}),
                    },
                ),
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------
def generate_session_title(llm: ChatOpenAI, user_q: str, assistant_a: str) -> str:
    prompt = (
        "다음 첫 질문과 답변을 한 줄로 요약하는 세션 제목을 한국어로 25자 이내로 작성하세요.\n"
        "따옴표, 설명, 번호 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{user_q[:1500]}\n\n[답변]\n{assistant_a[:2000]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = str(getattr(out, "content", out) or "").strip()
        title = title.strip("\"'")
        return title[:80] if title else "새 세션"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Title generation failed: %s", exc)
        return "새 세션"


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = remove_separators(str(getattr(out, "content", str(out)) or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _process_pdf_uploads(
    uploaded_files: list[Any],
    client: Client,
    session_id: str,
    user_id: str,
    embeddings: OpenAIEmbeddings,
    existing_files: set[str],
) -> tuple[list[str], int]:
    if not uploaded_files:
        return [], 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    new_names: list[str] = []
    total_chunks = 0

    for uf in uploaded_files:
        if uf.name in existing_files:
            continue
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata = dict(d.metadata or {})
                d.metadata["file_name"] = uf.name
            splits = splitter.split_documents(docs)
            if not splits:
                continue
            count = store_vectors_for_file(
                client, session_id, user_id, uf.name, splits, embeddings
            )
            total_chunks += count
            new_names.append(uf.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return new_names, total_chunks


# ---------------------------------------------------------------------------
# Session state & persistence
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults = {
        "chat_history": [],
        "conversation_memory": [],
        "active_session_id": None,
        "processed_names": [],
        "session_options": {},
        "selected_session_label": None,
        "logged_in_user_id": None,
        "logged_in_login_id": None,
        "_skip_selectbox_load": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _refresh_session_options(client: Client, user_id: str) -> None:
    sessions = fetch_sessions(client, user_id)
    st.session_state.session_options = {
        f"{s['title']} ({str(s['id'])[:8]})": str(s["id"]) for s in sessions
    }


def _first_qa_pair(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    user_q = ""
    for m in messages:
        if m["role"] == "user" and not user_q:
            user_q = m["content"]
        elif m["role"] == "assistant" and user_q:
            return user_q, m["content"]
    return None


def _ensure_active_session(
    client: Client,
    user_id: str,
    llm: ChatOpenAI | None = None,
) -> str:
    if st.session_state.active_session_id:
        sid = st.session_state.active_session_id
        if _session_belongs_to_user(client, sid, user_id):
            return sid
        st.session_state.active_session_id = None

    title = "새 세션"
    pair = _first_qa_pair(st.session_state.chat_history)
    if pair and llm:
        title = generate_session_title(llm, pair[0], pair[1])

    sid = create_session_row(
        client,
        user_id,
        title,
        file_names=list(st.session_state.processed_names),
    )
    st.session_state.active_session_id = sid
    if st.session_state.conversation_memory:
        persist_messages(client, sid, user_id, st.session_state.conversation_memory)
    return sid


def auto_save_session(client: Client, user_id: str) -> None:
    if not st.session_state.conversation_memory and not st.session_state.processed_names:
        return
    try:
        llm = get_llm()
    except ValueError:
        llm = None

    sid = st.session_state.active_session_id
    if not sid or not _session_belongs_to_user(client, sid, user_id):
        sid = _ensure_active_session(client, user_id, llm)
    else:
        pair = _first_qa_pair(st.session_state.chat_history)
        if pair and llm:
            title = generate_session_title(llm, pair[0], pair[1])
            update_session_meta(client, sid, user_id, title=title)

    persist_messages(client, sid, user_id, st.session_state.conversation_memory)
    update_session_meta(
        client,
        sid,
        user_id,
        file_names=list(st.session_state.processed_names),
    )
    _refresh_session_options(client, user_id)


def load_session_into_ui(client: Client, session_id: str, user_id: str) -> None:
    if not _session_belongs_to_user(client, session_id, user_id):
        st.error("선택한 세션에 접근할 수 없습니다.")
        return

    messages = fetch_messages(client, session_id, user_id)
    resp = (
        client.table("chat_sessions")
        .select("title, file_names")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    row = (resp.data or [{}])[0]
    file_names = row.get("file_names") or []
    if isinstance(file_names, str):
        try:
            file_names = json.loads(file_names)
        except json.JSONDecodeError:
            file_names = []

    st.session_state.active_session_id = session_id
    st.session_state.chat_history = list(messages)
    st.session_state.conversation_memory = list(messages)
    st.session_state.processed_names = list(file_names)
    st.session_state._skip_selectbox_load = True

    for lab, sid in st.session_state.session_options.items():
        if sid == session_id:
            st.session_state.selected_session_label = lab
            break


def insert_new_saved_session(client: Client, user_id: str) -> None:
    if not st.session_state.chat_history:
        st.warning("저장할 대화가 없습니다.")
        return

    llm = get_llm()
    pair = _first_qa_pair(st.session_state.chat_history)
    if pair:
        title = generate_session_title(llm, pair[0], pair[1])
    else:
        title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    new_id = create_session_row(
        client,
        user_id,
        title,
        file_names=list(st.session_state.processed_names),
    )
    persist_messages(client, new_id, user_id, st.session_state.conversation_memory)

    old_id = st.session_state.active_session_id
    if old_id and _session_belongs_to_user(client, old_id, user_id):
        copy_vectors_to_session(client, old_id, new_id, user_id)

    st.session_state.active_session_id = new_id
    _refresh_session_options(client, user_id)
    for lab, sid in st.session_state.session_options.items():
        if sid == new_id:
            st.session_state.selected_session_label = lab
            break
    st.success(f"세션이 저장되었습니다: {title}")


def clear_screen_only() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.active_session_id = None
    st.session_state.processed_names = []
    st.session_state.selected_session_label = None


def _on_session_select_change(client: Client, user_id: str) -> None:
    if st.session_state.get("_skip_selectbox_load"):
        st.session_state._skip_selectbox_load = False
        return
    label = st.session_state.get("selected_session_label")
    if not label or label == "(저장된 세션 없음)":
        return
    sid = st.session_state.session_options.get(label)
    if sid:
        load_session_into_ui(client, sid, user_id)


def _render_header() -> None:
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">기획예산처</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def _render_auth_panel(client: Client) -> None:
    st.info("로그인 후 챗봇과 세션 저장 기능을 사용할 수 있습니다.")
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        login_id = st.text_input("아이디", key="auth_login_id")
        password = st.text_input("비밀번호", type="password", key="auth_login_pw")
        if st.button("로그인", type="primary", key="btn_login"):
            uid, msg = login_user(client, login_id, password)
            if uid:
                st.session_state.logged_in_user_id = uid
                st.session_state.logged_in_login_id = login_id.strip()
                st.session_state.session_options = {}
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        new_id = st.text_input("새 아이디", key="auth_signup_id")
        new_pw = st.text_input("비밀번호", type="password", key="auth_signup_pw")
        new_pw2 = st.text_input("비밀번호 확인", type="password", key="auth_signup_pw2")
        if st.button("회원가입", key="btn_signup"):
            if new_pw != new_pw2:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                ok, msg = register_user(client, new_id, new_pw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def _render_chat_app(client: Client, user_id: str) -> None:
    login_label = st.session_state.get("logged_in_login_id") or "사용자"

    if not st.session_state.session_options:
        _refresh_session_options(client, user_id)

    with st.sidebar:
        st.markdown(f"**로그인:** `{login_label}`")
        if st.button("로그아웃"):
            logout_user()
            st.rerun()

        st.markdown(f"**모델:** `{MODEL_NAME}`")

        labels = list(st.session_state.session_options.keys()) or ["(저장된 세션 없음)"]
        if st.session_state.selected_session_label not in labels:
            st.session_state.selected_session_label = labels[0] if labels else None

        st.selectbox(
            "세션 선택",
            labels,
            key="selected_session_label",
            on_change=_on_session_select_change,
            args=(client, user_id),
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("세션저장"):
                try:
                    insert_new_saved_session(client, user_id)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"세션 저장 실패: {exc}")
            if st.button("세션로드"):
                label = st.session_state.get("selected_session_label")
                sid = st.session_state.session_options.get(label or "")
                if not sid:
                    st.warning("로드할 세션을 선택하세요.")
                else:
                    try:
                        load_session_into_ui(client, sid, user_id)
                        st.success("세션을 불러왔습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 로드 실패: {exc}")
        with col2:
            if st.button("세션삭제"):
                sid = st.session_state.active_session_id
                label = st.session_state.get("selected_session_label")
                if not sid and label:
                    sid = st.session_state.session_options.get(label)
                if not sid:
                    st.warning("삭제할 세션을 선택하세요.")
                else:
                    try:
                        delete_session_row(client, sid, user_id)
                        if st.session_state.active_session_id == sid:
                            clear_screen_only()
                        _refresh_session_options(client, user_id)
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 삭제 실패: {exc}")
            if st.button("화면초기화"):
                clear_screen_only()
                st.rerun()

        if st.button("vectordb"):
            sid = st.session_state.active_session_id
            if not sid:
                label = st.session_state.get("selected_session_label")
                sid = st.session_state.session_options.get(label or "")
            if not sid:
                st.warning("활성 세션이 없습니다. 세션을 선택하거나 대화를 시작하세요.")
            else:
                names = list_vector_file_names(client, sid, user_id)
                if names:
                    st.markdown("**Vector DB 파일 목록**")
                    for n in names:
                        st.text(f"- {n}")
                else:
                    st.info("이 세션에 저장된 벡터 파일이 없습니다.")

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    embeddings = get_embeddings()
                    sid = _ensure_active_session(client, user_id, get_llm())
                    existing = set(list_vector_file_names(client, sid, user_id))
                    new_names, chunk_count = _process_pdf_uploads(
                        list(uploads),
                        client,
                        sid,
                        user_id,
                        embeddings,
                        existing,
                    )
                    for n in new_names:
                        if n not in st.session_state.processed_names:
                            st.session_state.processed_names.append(n)
                    update_session_meta(
                        client,
                        sid,
                        user_id,
                        file_names=list(st.session_state.processed_names),
                    )
                    auto_save_session(client, user_id)
                    st.success(
                        f"PDF 처리 완료 (신규 파일 {len(new_names)}개, 청크 {chunk_count}개). 자동 저장되었습니다."
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        mem_count = len(st.session_state.conversation_memory)
        sid = st.session_state.active_session_id or "(없음)"
        st.text(
            f"모델: {MODEL_NAME}\n"
            f"활성 세션: {str(sid)[:8] if sid != '(없음)' else sid}\n"
            f"처리된 PDF: {len(st.session_state.processed_names)}\n"
            f"대화 메시지 수: {mem_count}"
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            llm = get_llm()
            embeddings = get_embeddings()
            sid = _ensure_active_session(client, user_id, llm)

            has_vectors = bool(list_vector_file_names(client, sid, user_id))
            if has_vectors:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                docs = retrieve_documents(
                    client, sid, user_id, user_input, embeddings, k=10
                )
                context = "\n\n".join(d.page_content for d in docs)
                messages = _build_rag_messages(user_input, context, mem_txt)
            else:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                messages = [SystemMessage(content=sys), HumanMessage(content=user_input)]

            acc = ""
            for chunk in llm.stream(messages):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    acc += piece
                    placeholder.markdown(remove_separators(acc) + "▌")
            full_answer = remove_separators(acc)
            placeholder.markdown(full_answer)

            if has_vectors and full_answer and not full_answer.lstrip().startswith("# 오류"):
                follow = _generate_followup_section(llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        st.session_state.conversation_memory.append(
            {"role": "assistant", "content": full_answer}
        )
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

        try:
            auto_save_session(client, user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("자동 저장 실패: %s", exc)


def main() -> None:
    st.set_page_config(
        page_title="기획예산처 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    _init_session()
    _apply_secrets_to_env()
    _render_header()

    missing = check_env_keys()
    if missing:
        hint = (
            "Streamlit Cloud에서는 **Settings → Secrets**에, "
            "로컬에서는 `.env`에 다음 키를 설정해 주세요."
        )
        st.error(
            f"다음 환경 변수가 설정되어 있지 않습니다: {', '.join(missing)}\n\n"
            f"{hint}\n\n로컬 `.env` 경로: `{ENV_PATH}`"
        )
        return

    client = get_supabase_client()
    if client is None:
        st.error("Supabase 클라이언트를 초기화할 수 없습니다.")
        return

    user_id = get_logged_in_user_id()
    if not user_id:
        _render_auth_panel(client)
        return

    _render_chat_app(client, user_id)


if __name__ == "__main__":
    main()
