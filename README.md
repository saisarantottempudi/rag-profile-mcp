# rag-profile-mcp

An MCP server that embeds your personal knowledge base (Brain wiki, CV, projects) into a local ChromaDB vector store and lets Claude answer questions about your skills, match job descriptions, and surface relevant project experience — all from natural language.

## Tools

| Tool | What it does |
|------|-------------|
| `index_wiki` | Embeds all wiki markdown into ChromaDB (run once) |
| `search_profile` | Semantic search across wiki with optional type filter |
| `match_jd` | Given a JD, returns ranked skill + project matches |
| `get_skill_summary` | Full content of a specific skill page |
| `list_projects` | List all projects, optionally filter by status |
| `index_status` | Show chunk count and type breakdown |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `BRAIN_WIKI_DIR` to your wiki path.

## Claude Code Integration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "rag-profile": {
      "command": "/Users/yourname/projects/rag-profile-mcp/venv/bin/python",
      "args": ["/Users/yourname/projects/rag-profile-mcp/server.py"],
      "env": {
        "BRAIN_WIKI_DIR": "/Users/yourname/Brain/wiki",
        "CHROMA_PATH": "/Users/yourname/.rag-profile-mcp/chroma"
      }
    }
  }
}
```

## Example Queries

- *"What skills do I have that match a senior data engineer role at Spotify?"*
- *"List all my complete projects and summarise what each one demonstrates"*
- *"Given this JD [paste], what are my strongest matches and biggest gaps?"*
- *"What do I know about transformer architectures?"*

## Stack

- **FastMCP** — MCP server framework
- **ChromaDB** — local vector store (persistent, no cloud)
- **sentence-transformers** (`all-MiniLM-L6-v2`) — fast local embeddings
