import os
import re
import gc
import torch
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# SemanticRAG
class SemanticRAG:
    def __init__(self, index_path, metadata_csv_path, embedding_model_name):
        self.index = faiss.read_index(index_path)
        self.df = pd.read_csv(metadata_csv_path)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def retrieve(self, query: str, top_filter_k: int = 30, top_retrieve_k: int = 5) -> pd.DataFrame:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

        equipment_name_match = re.search(r"\b(\d{4}[A-Z]{2})\b", query)
        equipment_name = equipment_name_match.group(1).strip() if equipment_name_match else ""

        if equipment_name:
            base_equipment_name = equipment_name.split('-')[0]
            filtered_df = self.df[self.df["equipment_name"].str.startswith(base_equipment_name)].copy()
            if filtered_df.empty:
                D_filter, I_filter = self.index.search(query_embedding, top_filter_k)
                filtered_df = self.df.iloc[I_filter[0]].copy()
        else:
            D_filter, I_filter = self.index.search(query_embedding, top_filter_k)
            filtered_df = self.df.iloc[I_filter[0]].copy()

        filtered_chunk_indices = filtered_df.index.tolist()
        filtered_chunk_embeddings = self.index.reconstruct_n(filtered_chunk_indices[0], len(filtered_chunk_indices))
        temp_index = faiss.IndexFlatL2(filtered_chunk_embeddings.shape[1])
        temp_index.add(filtered_chunk_embeddings)
        D_topic, I_topic_local = temp_index.search(query_embedding, top_retrieve_k)
        I_topic = [filtered_chunk_indices[i] for i in I_topic_local[0]]

        final_results = self.df.iloc[I_topic].copy()
        final_results["score"] = D_topic[0]

        return final_results[["equipment_name", "topic", "chunk_text", "score"]]


class AnswerQuestion:
    def __init__(self, retriever: SemanticRAG):
        model_path = os.environ.get("MODEL_PATH", "")
        max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

        gc.collect()
        torch.cuda.empty_cache()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",        # ✅ auto place on GPU/CPU
            low_cpu_mem_usage=False     # ✅ avoid meta tensors
        )

        

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.retriever = retriever
        self.max_new_tokens = max_new_tokens

    def format_paragraphs(self, text: str) -> str:
        return re.sub(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", "\n\n", text)

    def build_context(self, hits: pd.DataFrame, top_n: int = 3) -> str:
        top_chunks = hits.head(top_n)
        return "\n\n".join(
            [f"**{row['topic']}**: {row['chunk_text']}" for _, row in top_chunks.iterrows()]
        )

    def build_prompt(self, context: str, query: str) -> str:
      return f"""You are an AI assistant that answers questions strictly based on the provided context.
Do not use any outside knowledge. If the answer cannot be found in the context,
reply exactly with: "The information required to answer this question is not available in the provided context."

Follow these instructions carefully:
- Use clear, correct grammar and spelling.
- Be concise and factual.
- Do not mention the word "context" in your answer.

[EXAMPLE]
Context:
The cat is on the mat.

Question:
Where is the cat?

Answer:
The cat is on the mat.

Now, use the same pattern for the following input.

[CONTEXT]
{context}

[QUESTION]
{query}

[ANSWER]
"""


    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=temperature,
        )
        response = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return self.format_paragraphs(response.strip())

    def answer(self, query: str, top_n: int = 3, temperature: float = 0.7, max_new_tokens: int = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        hits = self.retriever.retrieve(query)
        context = self.build_context(hits, top_n=top_n)
        prompt = self.build_prompt(context, query)
        return self.generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
