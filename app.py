import os
import json
import uuid
import datetime as dt
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from ingest import retrieve

# ==========================
# CONFIG
# ==========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_PASSWORD = st.secrets["APP_PASSWORD"]
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
LOG_DIR = os.getenv("LOG_DIR", "./logs")

os.makedirs(LOG_DIR, exist_ok=True)

# ==========================
# CLONE CHROMA DB (PRIVATE REPO)
# ==========================
import subprocess

def clone_chroma():
    if os.path.exists(CHROMA_PATH):
        return  # ya existe

    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo_url = f"https://{token}@github.com/sebasgk8/chroma-db-private.git"

        subprocess.run(
            ["git", "clone", repo_url, CHROMA_PATH],
            check=True
        )

        st.write("✅ Chroma DB clonado desde repo privado")

    except Exception as e:
        st.error(f"❌ Error clonando Chroma DB: {e}")

clone_chroma()

# ==========================
# INIT
# ==========================
client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large"
)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

llm = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# AUTO INGEST (MVP FIX)
# ==========================
from ingest import ingest
import os

DATA_PATH = "./data"  # asegúrate de que existe en tu repo

def ensure_db():
    try:
        count = collection.count()
        st.write(f"📦 Chroma count inicial: {count}")

        if count == 0:
            st.warning("⚠️ Chroma vacío, ejecutando ingest...")

            if not os.path.exists(DATA_PATH):
                st.error(f"No existe carpeta de datos: {DATA_PATH}")
                return

            files = [f for f in os.listdir(DATA_PATH) if f.endswith((".pdf", ".docx"))]

            if not files:
                st.error("No hay documentos en /data")
                return

            for f in files:
                st.write(f"📄 Ingestando: {f}")
                ingest(os.path.join(DATA_PATH, f))

            st.success("✅ Ingest completado")

    except Exception as e:
        st.error(f"Ingest error: {e}")


ensure_db()

# ==========================
# DEBUG CHROMA
# ==========================
st.sidebar.write("📦 COUNT:", collection.count())

st.write(client.list_collections())

import os
st.sidebar.write("📁 EXISTS:", os.path.exists(CHROMA_PATH))

if os.path.exists(CHROMA_PATH):
    st.sidebar.write("📂 FILES:", os.listdir(CHROMA_PATH)[:5])

# ==========================
# SESSION ID (CLAVE NUEVA)
# ==========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id

# ==========================
# AUTH
# ==========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Password", type="password")

    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.stop()

# ==========================
# UTILS
# ==========================
def now_iso():
    return dt.datetime.utcnow().isoformat()


def log_event(data: dict):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        file = os.path.join(LOG_DIR, f"conversations_{dt.date.today().isoformat()}.jsonl")

        if "session_id" not in data:
            data["session_id"] = SESSION_ID

        with open(file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[LOG ERROR] {e}")


def is_greeting(text: str) -> bool:
    greetings = ["hola", "hello", "hi", "qué tal", "como estas", "cómo estás", "buenas"]
    t = text.lower()
    return any(g in t for g in greetings)

# ==========================
# MEMORY (LAST 5 TURNS)
# ==========================
def build_history_context(history, max_turns=5):
    recent = history[-max_turns:]

    context = ""
    for q, a, _, _, _ in recent:
        if a and len(a) > 20:
            context += f"User: {q}\nAssistant: {a}\n\n"

    return context

# ==========================
# CONTEXT LIMIT (NUEVO)
# ==========================
def build_context(chunks, max_chars=4000):
    context = ""
    for ch in chunks:
        if len(context) + len(ch) > max_chars:
            break
        context += ch + "\n\n"
    return context

# ==========================
# PROMPT
# ==========================
def build_prompt(query, context):
    return f"""
You are an enterprise assistant.

STRICT RULES:
- Answer ONLY using the provided context
- If the answer is not in the context, say exactly: "No sé"
- Be deterministic: same question → same answer
- Do not invent information
- Always answer in the same language as the question
- Provide detailed answers with examples when possible
- Always end with a follow-up question or suggestion
- If greeting, respond positively

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

# ==========================
# STREAMLIT UI
# ==========================
st.title("Chatbot Empresa")

if "history" not in st.session_state:
    st.session_state.history = []
if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}

for (q, a, chunks, metas, iid) in st.session_state.get("history", []):
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        st.markdown(a)

        col1, col2, col3 = st.columns([1, 1, 10])

        feedback = st.session_state.get("feedback_state", {}).get(iid)

        # 👍
        with col1:
            if st.button(
                "👍",
                key=f"up_{iid}",
                disabled=feedback is not None
            ):
                st.session_state.feedback_state[iid] = "up"

                log_event({
                    "timestamp": now_iso(),
                    "interaction_id": iid,
                    "feedback": "up"
                })

        # 👎
        with col2:
            if st.button(
                "👎",
                key=f"down_{iid}",
                disabled=feedback is not None
            ):
                st.session_state.feedback_state[iid] = "down"

                log_event({
                    "timestamp": now_iso(),
                    "interaction_id": iid,
                    "feedback": "down"
                })

        with col3:
            if feedback == "up":
                st.markdown("<span style='color:gray'>✔ Feedback positivo</span>", unsafe_allow_html=True)
            elif feedback == "down":
                st.markdown("<span style='color:gray'>✖ Feedback negativo</span>", unsafe_allow_html=True)


query = st.chat_input("Haz tu pregunta...")

if query:
    st.session_state.history.append((query, "", [], [], None))

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Pensando..."):

            chunks = retrieve(query)
            chunks_meta = []

            response = llm.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0,
                top_p=0,
                messages=[{
                    "role": "user",
                    "content": build_prompt(
                        query,
                        f"CONVERSATION HISTORY:\n{build_history_context(st.session_state.history)}\n\nDOCUMENT CONTEXT:\n{build_context(chunks)}"
                    )
                }],
                stream=True
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)

                if token:
                    full_response += token
                    placeholder.markdown(full_response)

        if not full_response:
            full_response = "No sé. ¿Puedes reformular la pregunta?"

    interaction_id = str(uuid.uuid4())

    st.session_state.history[-1] = (
        query,
        full_response,
        chunks,
        chunks_meta,
        interaction_id
    )

    log_event({
        "timestamp": now_iso(),
        "interaction_id": interaction_id,
        "session_id": SESSION_ID,
        "query": query,
        "answer": full_response,
        "chunks": chunks,
        "chunks_meta": chunks_meta,
        "feedback": None
    })

    st.rerun()