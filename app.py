import re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

load_dotenv()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ---------- YouTube ID extraction ----------
def extract_video_id(url: str) -> str | None:
    url = url.strip()
    if "youtu.be" in url:
        path = urlparse(url).path
        match = re.search(r"/([a-zA-Z0-9_-]{11})", path)
        if match:
            return match.group(1)
    elif re.search(r"/(embed|shorts|live)/([a-zA-Z0-9_-]{11})", url):
        match = re.search(r"/(?:embed|shorts|live)/([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    return None

# ---------- Build / retrieve vector store (CACHED) ----------
@st.cache_resource(show_spinner=False)
def get_vector_store(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        return None
    except Exception:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = get_embeddings()
    return FAISS.from_documents(documents=chunks, embedding=embeddings)

# ---------- Ask a question ----------
def ask_question(video_id: str, question: str, chat_history: list) -> str:
    vector_store = get_vector_store(video_id)
    if vector_store is None:
        reply = "No captions available for this video."
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=reply))
        return reply

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer ONLY from the transcript context below.\n"
                "If the answer is not present, say \"I don't know\".\n\n"
                f"Transcript Context:\n{context}"
            )
        ),
        *chat_history,
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))
    return response.content

# ====================== STREAMLIT APP ======================
st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"   # will still try, but not guaranteed
)

# ---------- State initialisation ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

# ---------- Handle URL query parameters ----------
query_params = st.query_params
if "video_id" in query_params and query_params["video_id"]:
    url_video_id = query_params["video_id"]
    if url_video_id != st.session_state.current_video_id:
        st.session_state.current_video_id = url_video_id
        st.session_state.messages = []
        st.query_params.clear()

# ---------- Sidebar (always contains the input) ----------
with st.sidebar:
    st.title("🎬 YouTube Chat")

    youtube_url = st.text_input(
        "Paste YouTube Link",
        value=f"https://www.youtube.com/watch?v={st.session_state.current_video_id}"
        if st.session_state.current_video_id
        else "",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id is None:
            st.error("❌ Could not extract a valid YouTube ID from the link.")
            # Do not stop the whole app – we still want the main area to show instructions if needed
        else:
            if video_id != st.session_state.current_video_id:
                st.session_state.current_video_id = video_id
                st.session_state.messages = []

            vector_store = get_vector_store(video_id)
            if vector_store is None:
                st.error("⚠️ No captions available for this video.")
            else:
                st.video(f"https://www.youtube.com/watch?v={video_id}")
                st.caption(f"Video ID: `{video_id}`")

                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    st.rerun()
    else:
        st.info("Enter a YouTube link above")

# ---------- Main area ----------
st.header("💬 Chat with the video")

# If no video is loaded, show the instructional landing page
if st.session_state.current_video_id is None:
    st.markdown("""
    ### 👈 Please open the sidebar first!
    Click the **arrow** or **hamburger menu** (☰) in the top‑left corner to open the sidebar.
    Then paste a YouTube link to start chatting with the video.
    """)
    st.stop()   # hide the rest (chat history, input) until a video is loaded

# Video is loaded – display conversation and chat input
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

question = st.chat_input("Ask something about the video...")

if question:
    with st.spinner("Thinking..."):
        answer = ask_question(
            video_id=st.session_state.current_video_id,
            question=question,
            chat_history=st.session_state.messages,
        )
    st.rerun()