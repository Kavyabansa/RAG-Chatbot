# Kavya's ML Journey RAG Chatbot — Streamlit

This is the Streamlit version of the uploaded `week9_day6_rag_chatbot.ipynb`.

## Architecture

- Streamlit — web UI
- Sentence Transformers (`all-MiniLM-L6-v2`) — embeddings
- FAISS — vector search
- LangChain — prompt + conversation handling
- Groq — LLM generation
- RAG — retrieves the top 3 relevant ML-journey documents before answering

## Run locally

1. Install Python.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your Groq API key.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Or create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

5. Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
3. Go to Streamlit Community Cloud and create a new app.
4. Select your GitHub repository.
5. Set the main file to `app.py`.
6. Deploy.
7. Open the app's Settings → Secrets.
8. Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Do NOT upload your real API key to GitHub.

## Important

The original notebook contained a hard-coded Groq API key. Do not reuse or publish that exposed key. If it is a real active key, revoke/rotate it in Groq and create a new key for this Streamlit deployment.
