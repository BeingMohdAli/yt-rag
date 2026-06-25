# 🎬 Ask the Video
Problem
Watching a YouTube video to find one specific piece of information is slow. A 40-minute tutorial, lecture, or podcast might contain the answer to your question in just two minutes — but there's no way to know where without watching the whole thing, scrubbing through the timeline, or hoping the creator added timestamps.
Manually skimming auto-generated captions isn't much better: they're a wall of unstructured text with no search, no context, and no way to ask a follow-up question.
YT RAG Chat solves this by letting you paste any YouTube link and just ask it questions, like you would a person who already watched the video. It pulls the video's transcript, breaks it into searchable chunks, and uses retrieval-augmented generation (RAG) to find the most relevant part of the transcript for your question — then has an LLM (Google Gemini) answer based only on that context, so responses stay grounded in what was actually said in the video rather than the model's general knowledge.
This turns any YouTube video into something you can interrogate conversationally instead of watching passively.
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
