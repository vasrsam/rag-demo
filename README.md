🧠 **Project Overview**

Objective:
This project demonstrates the application of Retrieval-Augmented Generation (RAG) for building internal enterprise chatbots as part of an Enterprise Information Management (EIM) system.
The demonstration use case focuses on engineering equipment maintenance manuals.

⚙️ **Requirements**

The application is built using Streamlit.

It requires a GPU for efficient inference.

For demonstration purposes, the app runs on Google Colab with an A100 GPU.

🧩 **Data Preprocessing**

The maintenance manuals are available as machine-readable PDFs, so OCR is not required.

Text is chunked based on document subheadings, and metadata (equipment name) is added to each chunk to enable metadata-based filtering during retrieval.

spaCy and regular expressions (regex) are used to clean and normalize text.

Cleaned text chunks are embedded and indexed using FAISS for fast semantic retrieval.

Both the embedded index and the metadata-enriched text chunks are stored for use during query-time retrieval.

🔍 **Methodology**

Query Embedding:
The user query is converted into a vector representation using an embedding model.

Context Retrieval:
Relevant document chunks are retrieved based on:

Equipment name (metadata filtering), and

Semantic similarity with the query embedding.

Response Generation:
Retrieved chunks are passed to a Large Language Model (LLM) to generate the final response.

Note: The prompt is carefully engineered to:

Keep responses strictly within the retrieved context,

Prevent answers outside the document scope, and

Minimize hallucination by restricting the model from using external knowledge when a query is outside the domain.

<img width="1786" height="980" alt="image" src="https://github.com/user-attachments/assets/d0802cf4-1c3f-4027-9e3d-f0a8b29f6e66" />

**Models used**:

*Embedding model* used for encoding chunks and query: all-MiniLM-L6-v2

*LLM used* for answering query: mistral-7b-instruct-v0.1


