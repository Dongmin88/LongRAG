import os
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Document:
    content: str
    score: float = 0.0

class Config:
    def __init__(self):
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base")
        self.llm_model = os.getenv("LLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        
        if not self.huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

class Retriever:
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.embedding_model,
            token=config.huggingface_token
        )
        self.model = AutoModel.from_pretrained(
            config.embedding_model,
            token=config.huggingface_token
        )
        
    def _embed(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings
    
    def retrieve(self, query: str, corpus: List[str], k: int = 5) -> List[Document]:
        query_emb = self._embed([query])
        doc_embs = self._embed(corpus)
        
        scores = torch.matmul(query_emb, doc_embs.T).squeeze()
        top_k_scores, top_k_indices = torch.topk(scores, min(k, len(corpus)))
        
        return [
            Document(content=corpus[idx], score=score.item())
            for score, idx in zip(top_k_scores, top_k_indices)
        ]

class ReorderingStrategy:
    @staticmethod
    def reorder_documents(documents: List[Document]) -> List[Document]:
        n = len(documents)
        reordered = [None] * n
        
        for i in range(n):
            if i % 2 == 0:
                pos = (i + 1) // 2
            else:
                pos = n - (i // 2)
            reordered[pos-1] = documents[i]
            
        return reordered

class RAG:
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
            
        self.config = config
        self.retriever = Retriever(config)
        
        # Initialize LLM with Llama 2
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-1b-chat-hf",  # Using 1B version
            token=config.huggingface_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-1b-chat-hf",  # Using 1B version
            token=config.huggingface_token,
            device_map="auto"  # Automatically handle device placement
        )
        
        self.reorderer = ReorderingStrategy()
        
    def _create_prompt(self, query: str, documents: List[Document]) -> str:
        # Llama 2 specific prompt format
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.content}" 
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""[INST] Based on the following documents, please answer this question:
{query}

Here are the relevant documents:
{docs_text}
[/INST]"""
        
        return prompt
    
    def generate(self, 
                query: str,
                corpus: List[str],
                num_docs: int = 5,
                max_new_tokens: int = 256) -> str:
        
        retrieved_docs = self.retriever.retrieve(query, corpus, k=num_docs)
        reordered_docs = self.reorderer.reorder_documents(retrieved_docs)
        prompt = self._create_prompt(query, reordered_docs)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up response by removing the instruction part
        response = response.split("[/INST]")[-1].strip()
        return response

def main():
    # Sample usage
    
    # First, set up your API key in a .env file:
    # HUGGINGFACE_TOKEN=your_token_here
    
    # Initialize config
    config = Config()
    
    # Sample corpus
    corpus = [
        "The capital of France is Paris.",
        "Paris is known for the Eiffel Tower.",
        "The Eiffel Tower was completed in 1889.",
        "France has a population of about 67 million.",
        "French is the official language of France."
    ]
    
    # Initialize RAG
    rag = RAG(config)
    
    # Example query
    query = "What is the capital of France and what is it known for?"
    
    try:
        # Generate response
        response = rag.generate(query, corpus)
        print(f"Query: {query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()