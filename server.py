"""
RAG Profile Assistant MCP
Embeds Brain wiki + CV into ChromaDB, answers questions about skills,
projects, and generates JD-matched cover letters / skill summaries.
"""

import os
import glob
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from fastmcp import FastMCP

# ── Config ─────────────────────────────────────────────────────────────────
BRAIN_WIKI_DIR = os.environ.get("BRAIN_WIKI_DIR", str(Path.home() / "Brain" / "wiki"))
CHROMA_PATH    = os.environ.get("CHROMA_PATH",    str(Path.home() / ".rag-profile-mcp" / "chroma"))
EMBED_MODEL    = os.environ.get("EMBED_MODEL",    "all-MiniLM-L6-v2")
COLLECTION     = "brain_wiki"

# ── Chroma setup (lazy — avoid loading the embedding model at import time) ──
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

_ef = None
_client = None

def _get_collection():
    global _ef, _client
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client.get_or_create_collection(name=COLLECTION, embedding_function=_ef)

def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return chunks

# ── MCP ────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "rag-profile-assistant",
    instructions=(
        "RAG assistant over Sai Saran's Brain wiki. "
        "Use `index_wiki` once (or when wiki changes) then query freely."
    ),
)


@mcp.tool()
def index_wiki(force_reindex: bool = False) -> str:
    """
    Embed all markdown pages from Brain wiki into ChromaDB.
    Run once after setup, or with force_reindex=True after wiki updates.
    """
    col = _get_collection()

    if not force_reindex and col.count() > 0:
        return f"Already indexed {col.count()} chunks. Pass force_reindex=True to rebuild."

    if force_reindex:
        _client.delete_collection(COLLECTION)
        col = _client.get_or_create_collection(name=COLLECTION, embedding_function=_ef)

    md_files = glob.glob(os.path.join(BRAIN_WIKI_DIR, "**", "*.md"), recursive=True)
    if not md_files:
        return f"No markdown files found in {BRAIN_WIKI_DIR}"

    docs, ids, metas = [], [], []
    for path in md_files:
        rel = os.path.relpath(path, BRAIN_WIKI_DIR)
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception:
            continue

        # derive page type from directory
        parts = rel.split(os.sep)
        page_type = parts[0] if len(parts) > 1 else "root"

        for idx, chunk in enumerate(_chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{rel}::{idx}")
            metas.append({"source": rel, "type": page_type})

    # batch upsert
    batch = 100
    for i in range(0, len(docs), batch):
        col.upsert(
            documents=docs[i : i + batch],
            ids=ids[i : i + batch],
            metadatas=metas[i : i + batch],
        )

    return f"Indexed {len(docs)} chunks from {len(md_files)} files."


@mcp.tool()
def search_profile(query: str, n_results: int = 5, page_type: Optional[str] = None) -> str:
    """
    Semantic search over the wiki. Returns relevant excerpts with source paths.
    page_type filter: skills | projects | career | concepts | companies | roles | personal | news | immigration
    """
    col = _get_collection()
    if col.count() == 0:
        return "Wiki not indexed yet. Run `index_wiki` first."

    where = {"type": page_type} if page_type else None
    results = col.query(query_texts=[query], n_results=min(n_results, col.count()), where=where)

    if not results["documents"][0]:
        return "No relevant chunks found."

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = round(1 - dist, 3)
        output.append(f"[{score}] {meta['source']}\n{doc[:400]}\n")

    return "\n---\n".join(output)


@mcp.tool()
def match_jd(job_description: str, top_k: int = 8) -> str:
    """
    Given a job description, find the most relevant skills and projects from the wiki.
    Returns a structured match report: what aligns, what gaps exist.
    """
    col = _get_collection()
    if col.count() == 0:
        return "Wiki not indexed yet. Run `index_wiki` first."

    # search skills and projects separately for richer coverage
    skill_results = col.query(
        query_texts=[job_description],
        n_results=min(top_k, col.count()),
        where={"type": "skills"},
    )
    project_results = col.query(
        query_texts=[job_description],
        n_results=min(top_k, col.count()),
        where={"type": "projects"},
    )

    def fmt(results):
        out = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = round(1 - dist, 3)
            out.append(f"  [{score}] {meta['source']}\n  {doc[:300]}")
        return "\n".join(out) if out else "  None found."

    return (
        f"=== SKILLS MATCH ===\n{fmt(skill_results)}\n\n"
        f"=== PROJECTS MATCH ===\n{fmt(project_results)}\n\n"
        "Use these results to write a targeted cover letter or identify gaps."
    )


@mcp.tool()
def get_skill_summary(skill_name: str) -> str:
    """
    Return full content of a specific skill page from the wiki.
    """
    pattern = os.path.join(BRAIN_WIKI_DIR, "skills", f"*{skill_name.lower()}*")
    matches = glob.glob(pattern)
    if not matches:
        # fallback: semantic search
        return search_profile(f"skill: {skill_name}", n_results=3, page_type="skills")
    return Path(matches[0]).read_text(encoding="utf-8")


@mcp.tool()
def list_projects(status: Optional[str] = None) -> str:
    """
    List all project wiki pages. Optionally filter by status: idea | in-progress | complete | shelved
    """
    proj_dir = Path(BRAIN_WIKI_DIR) / "projects"
    if not proj_dir.exists():
        return "No projects directory found."

    output = []
    for f in sorted(proj_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        # extract frontmatter status
        proj_status = "unknown"
        for line in text.splitlines():
            if line.startswith("status:"):
                proj_status = line.split(":", 1)[1].strip()
                break
        if status and proj_status != status:
            continue
        # extract name
        name = f.stem
        for line in text.splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
                break
        output.append(f"- {name} ({proj_status})  [{f.name}]")

    return "\n".join(output) if output else "No matching projects."


@mcp.tool()
def index_status() -> str:
    """Show how many chunks are indexed and which page types exist."""
    col = _get_collection()
    count = col.count()
    if count == 0:
        return "Nothing indexed. Run `index_wiki`."

    # sample to get type breakdown
    sample = col.get(limit=count, include=["metadatas"])
    type_counts: dict[str, int] = {}
    for m in sample["metadatas"]:
        t = m.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    breakdown = "\n".join(f"  {t}: {c} chunks" for t, c in sorted(type_counts.items()))
    return f"Total: {count} chunks\n\nBreakdown:\n{breakdown}"


if __name__ == "__main__":
    mcp.run()
