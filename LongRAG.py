import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Union
from dataclasses import dataclass
import json

@dataclass
class Document:
    content: str
    score: float = 0.0
    is_hard_negative: bool = False

class EnhancedRAG:
    def __init__(self, api_key: str):
        # Embedding model (e5-small for efficiency)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/e5-small",
            token=api_key
        )
        self.embed_model = AutoModel.from_pretrained(
            "intfloat/e5-small",
            token=api_key
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # LLama 3.2 1B
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

    def _compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Compute embeddings for given texts"""
        inputs = self.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.embed_model.device)
        
        with torch.no_grad():
            embeddings = self.embed_model(**inputs).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def retrieve(self, query: str, corpus: List[str], k: int = 5) -> List[Document]:
        """Enhanced retrieval with hard negatives detection"""
        query_emb = self._compute_embeddings([query])
        doc_embs = self._compute_embeddings(corpus)
        
        # Calculate similarities
        similarities = torch.matmul(query_emb, doc_embs.T).squeeze()
        
        # Get top-k documents
        top_k_scores, top_k_indices = torch.topk(similarities, min(k, len(corpus)))
        
        # Detect hard negatives (documents with high similarity but potentially misleading)
        similarity_threshold = similarities.mean() + similarities.std()
        is_hard_negative = similarities > similarity_threshold
        
        documents = []
        for idx, score in zip(top_k_indices.cpu(), top_k_scores.cpu()):
            idx = idx.item()
            documents.append(Document(
                content=corpus[idx],
                score=score.item(),
                is_hard_negative=is_hard_negative[idx].item()
            ))
        
        # Apply retrieval reordering
        return self._reorder_documents(documents)

    def _reorder_documents(self, documents: List[Document]) -> List[Document]:
        """Lost-in-the-middle mitigation through reordering"""
        n = len(documents)
        if n == 0:
            return []
            
        # Sort by score
        documents = sorted(documents, key=lambda x: x.score, reverse=True)
        
        # Reorder: important docs at start and end
        reordered = []
        left, right = 0, n-1
        is_left = True
        
        while left <= right:
            if is_left:
                reordered.append(documents[left])
                left += 1
            else:
                reordered.append(documents[right])
                right -= 1
            is_left = not is_left
            
        return reordered

    def generate_response(self, query: str, documents: List[Document]) -> Dict[str, str]:
        """Generate response with intermediate reasoning"""
        # Format documents text
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.content}" +
            (" [Hard Negative]" if doc.is_hard_negative else "")
            for i, doc in enumerate(documents)
        ])

        # First prompt for reasoning
        reasoning_prompt = f"""<|system|>You are a helpful AI assistant that first analyzes documents and then answers questions.</s>
<|user|>I need help analyzing these documents to answer a question.

Question: {query}

Documents:
{docs_text}

First, explain your reasoning about which documents are most relevant and why.</s>
<|assistant|>Let me analyze the documents and explain my reasoning:</s>"""

        # Generate reasoning
        reasoning_inputs = self.llm_tokenizer(reasoning_prompt, return_tensors="pt").to(self.llm_model.device)
        reasoning_outputs = self.llm_model.generate(
            **reasoning_inputs,
            max_new_tokens=200,
            num_beams=4,
            temperature=0.7,
            top_p=0.9
        )
        reasoning = self.llm_tokenizer.decode(reasoning_outputs[0], skip_special_tokens=True)
        reasoning = reasoning.split("Let me analyze")[-1].strip()

        # Second prompt for final answer
        answer_prompt = f"""<|system|>Now provide a final answer based on your analysis.</s>
<|user|>Based on your analysis:
{reasoning}

Please provide a clear and concise answer to the original question: {query}</s>
<|assistant|>Here's my answer based on the analysis:</s>"""

        # Generate final answer
        answer_inputs = self.llm_tokenizer(answer_prompt, return_tensors="pt").to(self.llm_model.device)
        answer_outputs = self.llm_model.generate(
            **answer_inputs,
            max_new_tokens=150,
            num_beams=4,
            temperature=0.7,
            top_p=0.9
        )
        
        final_answer = self.llm_tokenizer.decode(answer_outputs[0], skip_special_tokens=True)
        final_answer = final_answer.split("Here's my answer")[-1].strip()

        return {
            "reasoning": reasoning,
            "answer": final_answer
        }

    def process_query(self, query: str, corpus: List[str], k: int = 5) -> Dict[str, str]:
        """Complete RAG pipeline"""
        retrieved_docs = self.retrieve(query, corpus, k)
        return self.generate_response(query, retrieved_docs)

# 샘플 문서 데이터베이스 생성
def create_sample_database() -> List[str]:
    documents = [
        # AI & Technology
        "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. Modern AI systems use deep learning and neural networks.",
        "Machine Learning is a subset of AI that enables systems to learn and improve from experience without explicit programming.",
        "Deep Learning is part of machine learning based on artificial neural networks. It's particularly effective in image and speech recognition.",
        "Natural Language Processing (NLP) is a branch of AI focused on enabling computers to understand and process human language.",
        "The Transformer architecture, introduced in 2017, revolutionized NLP tasks through its attention mechanism.",
        
        # Science & Space
        "The James Webb Space Telescope is the most powerful space telescope ever built, launched in December 2021.",
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape from them.",
        "Quantum computing uses quantum mechanics principles to perform calculations exponentially faster than classical computers.",
        "Dark matter makes up about 85% of the matter in the universe, but it cannot be directly observed.",
        "The Large Hadron Collider is the world's largest and highest-energy particle collider.",
        
        # Climate & Environment
        "Global warming refers to the long-term rise in Earth's average temperature due to human activities.",
        "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
        "The Amazon rainforest produces about 20% of Earth's oxygen and is often called the 'lungs of the Earth'.",
        "Electric vehicles produce zero direct emissions, helping reduce air pollution in urban areas.",
        "Ocean acidification occurs when seawater absorbs too much CO2 from the atmosphere.",
        
        # Medicine & Health
        "MRNA vaccines teach our cells how to make a protein that triggers an immune response.",
        "Antibiotics are medications used to treat bacterial infections but are ineffective against viruses.",
        "CRISPR gene editing technology allows scientists to modify DNA sequences and alter gene function.",
        "The human brain contains approximately 86 billion neurons connected through trillions of synapses.",
        "Telemedicine enables healthcare providers to treat patients remotely using telecommunications technology."
    ]
    return documents

def main():
    # Hugging Face API 키 설정
    API_KEY = "your_api_key_here"
    
    try:
        # RAG 시스템 초기화
        rag = EnhancedRAG(api_key=API_KEY)
        
        # 샘플 문서 로드
        corpus = create_sample_database()
        
        # 테스트 쿼리
        query = "How does AI relate to machine learning and deep learning?"
        
        # 응답 생성
        response = rag.process_query(query, corpus, k=3)
        
        print("Query:", query)
        print("\nReasoning:", response["reasoning"])
        print("\nFinal Answer:", response["answer"])
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()