# 🎬 Ask the Video

A Retrieval-Augmented Generation (RAG) app that answers questions about any YouTube video — grounded in its actual captions, not guesses.

The Diagram
<img width="1053" height="579" alt="image" src="https://github.com/user-attachments/assets/7b193170-ed48-4152-80e2-9755c07914bb" />



## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| LLM | Google Gemini (`langchain-google-genai`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Transcripts | `youtube-transcript-api` |


> The first run downloads the MiniLM embedding model (a few hundred MB), so give it a minute.



## ⚠️ Limitations

- Only works on videos that have captions (auto-generated or manual) available.
- Answers are only as good as the captions themselves; auto-generated captions can contain transcription errors.

## 📄 License

MIT — feel free to use and adapt.


Connect me at
email = mohdalisaad868@gmail.com
linkedIn = https://www.linkedin.com/in/mohd-ali-529684378/
