**Objective**:
This project is to demonstrate the application of Retrieval Augmented Generation to build internal chatbots as part Enterprise Information Management
system. The use case selected in demonstration is Engineering equipment maintenance manuals. 

**Requirements**:
The application is built on streamlit and it needs GPU to run the application. For demonstration the application is run with Google colab (A100 GPU).

**Preprocessing**:
The maintenance manuals are in machine readable PDFs, thus OCR is not required to ingest the documents. Chunking of text is done based on subheading and metadata
about equipment name is added to the dataframe to enable filtering the chunks based on equipment name. Spacy and regex used to clean the text. The cleaned text chunks are 
embedded and indexed using FAISS. Chucked text with metadata also saved for retrieval.

**Methodology**:
1. Query is embedded using Embedding model
2. The relevant chuncks are filtered based on Equipment name and semantic similarity with query
3. The retrieved chuncks are passed to LLM for repsonse
(Note: Prompt engineering is done to ensure the Response is not outside the context in passed documents and if question is asked outside domain it doesn't
give answer from external learning (i.e minimum halluciation).
<img width="1786" height="980" alt="image" src="https://github.com/user-attachments/assets/d0802cf4-1c3f-4027-9e3d-f0a8b29f6e66" />

**Models used**:
Embedding model used for encoding chunks and query: all-MiniLM-L6-v2\n
LLM used for answering query: mistral-7b-instruct-v0.1


