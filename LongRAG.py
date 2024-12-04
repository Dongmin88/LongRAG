import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    score: float = 0.0

class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Initialize embedding model
        self.embed_tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/e5-small", 
            token=api_key
        )
        self.embed_model = AutoModel.from_pretrained(
            "intfloat/e5-small", 
            token=api_key
        )
        
        # Initialize Llama 3.2 1B
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            token=api_key
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            token=api_key,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    def retrieve(self, query: str, corpus: List[str], k: int = 5) -> List[Document]:
        inputs = self.embed_tokenizer(
            [query] + corpus,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            embeddings = self.embed_model(**inputs).last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            query_emb = embeddings[0].unsqueeze(0)
            doc_embs = embeddings[1:]
            
            similarities = torch.matmul(query_emb, doc_embs.T).squeeze()
            top_k_scores, top_k_indices = torch.topk(similarities, min(k, len(corpus)))
            
            documents = [
                Document(content=corpus[idx], score=score.item())
                for idx, score in zip(top_k_indices, top_k_scores)
            ]
            
            return self._reorder_documents(documents)
    
    def _reorder_documents(self, documents: List[Document]) -> List[Document]:
        n = len(documents)
        reordered = [None] * n
        
        for i in range(n):
            if i % 2 == 0:
                pos = (i + 1) // 2
            else:
                pos = n - (i // 2)
            reordered[pos-1] = documents[i]
            
        return reordered
    
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.content}" 
            for i, doc in enumerate(documents)
        ])
        
        # Llama 3 specific prompt format
        prompt = f"""<|system|>You are a helpful AI assistant that answers questions based on provided documents.</s>
<|user|>Answer this question using the documents below:

Question: {query}

Documents:
{docs_text}</s>
<|assistant|>I'll help answer your question based on the provided documents.</s>"""

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant's response
        return response.split("<|assistant|>")[-1].strip()
    
    def run_rag(self, query: str, corpus: List[str], k: int = 5) -> str:
        retrieved_docs = self.retrieve(query, corpus, k)
        answer = self.generate_answer(query, retrieved_docs)
        return answer

def main():
    # Your Hugging Face API key
    API_KEY = "your_huggingface_api_key_here"
    
    # Initialize RAG system
    rag = RAGSystem(api_key=API_KEY)
    
    # Example corpus
    corpus = [
        "The capital of France is Paris.",
        "Paris is known for the Eiffel Tower.",
        "The Eiffel Tower was completed in 1889.",
        "France has a population of about 67 million.",
        "French is the official language of France."
    ]
    
    # Example query
    query = "What is the capital of France and what is it known for?"
    
    try:
        answer = rag.run_rag(query, corpus)
        print("Query:", query)
        print("\nAnswer:", answer)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()