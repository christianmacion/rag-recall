# Deploy — rag-recall

**TL;DR:** a public URL in ~5 minutes on Streamlit Community Cloud (free). Works with **no API key** (offline stdlib TF-IDF retriever + deterministic answers); add `ANTHROPIC_API_KEY` for live LLM answers.

## 0. Prerequisites
- GitHub account + a Streamlit Community Cloud account ([share.streamlit.io](https://share.streamlit.io), sign in with GitHub — free).
- *(Optional, live mode only)* an Anthropic API key.

## 1. Own public GitHub repo
```bash
cd "06_projects/rag-recall"
git init && git add . && git commit -m "rag-recall: measured RAG with a retrieval scorecard"
gh repo create rag-recall --public --source=. --push
# or add the remote manually and push to main (see any sibling DEPLOY.md).
```

### (Recommended) lean requirements for a fast deploy
The app's **default retriever is stdlib TF-IDF**, so it does *not* need the heavy embedding libs. For the fastest cold start, comment out `sentence-transformers` and `chromadb` in `requirements.txt` before pushing (re-enable them only if you want the optional embedding retriever — it downloads an ~80 MB model on first load).

## 2. Deploy on Streamlit Community Cloud
1. [share.streamlit.io](https://share.streamlit.io) → **Create app** → **Deploy from GitHub**.
2. Repo `<you>/rag-recall` · Branch `main` · **Main file path: `app.py`**.
3. *(Optional, live answers)* **Advanced settings → Secrets**:
   ```toml
   ANTHROPIC_API_KEY="sk-ant-..."
   ```
   Streamlit exposes secrets as environment variables, so the app's `os.getenv("ANTHROPIC_API_KEY")` picks it up automatically.
4. **Deploy** → permanent URL `https://<app>.streamlit.app`.

## What a reviewer sees
Click **Run eval** → a scorecard: **recall@3 = 0.886, MRR@3 = 0.805** on 35 labeled questions, plus a **retrieval-failure gallery** (the misses, root-caused). Then ask their own question and see exactly which chunks were cited. Toggle `k` and chunk size and watch the metrics move.

## Run locally
```bash
pip install -r requirements.txt
python -m rag_recall.evaluate      # prints the scorecard, writes scorecard.json
streamlit run app.py
```

## Alternative host: Hugging Face Spaces (SDK: Streamlit). Add the key under **Settings → Variables and secrets** for live mode.

---
*Christian Macion — AI / Agent Engineer.*
