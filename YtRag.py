from langchain_google_genai import (
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

video_id = "_DppFitt_6E"

try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

except TranscriptsDisabled:
    print("No captions available for this video")
    exit()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 4,
        "lambda_mult": 0.5
    }
)



prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.
    {context}
    {question}
    """,
    input_variables=["context", "question"]
)

question = "which type of camera would be in damsung fold phone which looks similar to iphone"


retreived_docs = retriever.invoke(question)

context = "\n\n".join(doc.page_content for doc in retreived_docs)

final_prompt = prompt.invoke({"context":context, "question":question})

result = llm.invoke(final_prompt)

print(result.content)





