
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


# Project paths
POLICY_DIR = BASE_DIR / "data" / "policies"
EVAL_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"


# Part 1: Load and chunk policy documents
def load_policy_chunks(chunk_size=40, overlap=10):
    chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        words = file_path.read_text(encoding="utf-8").split()

        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "source": file_path.name,
                "text": chunk_text
            })

            # Stop after the final chunk
            if end >= len(words):
                break

            start += chunk_size - overlap

    return chunks


# Part 2: Build TF-IDF search
class PolicyRAG:
    def __init__(self):
        self.chunks = load_policy_chunks()

        self.vectorizer = TfidfVectorizer()

        self.chunk_texts = [
            chunk["text"] for chunk in self.chunks
        ]

        self.chunk_vectors = self.vectorizer.fit_transform(
            self.chunk_texts
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.chunk_vectors
        ).flatten()

        ranked_indices = similarities.argsort()[::-1][:k]

        results = []

        for index in ranked_indices:
            results.append({
                "source": self.chunks[index]["source"],
                "score": float(similarities[index]),
                "text": self.chunks[index]["text"]
            })

        return results

    # Part 3: Generate an answer using mock_llm
    def answer(self, question):
        retrieved = self.retrieve(question, k=3)

        context_parts = []

        for item in retrieved:
            context_parts.append(
                f"[{item['source']}]\n{item['text']}"
            )

        context = "\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "POLICY ASSISTANT. "
                    "Answer only using the supplied policy context. "
                    "If the answer is not available in the context, "
                    "say that you are unable to find it."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n"
                    f"QUESTION:\n{question}"
                )
            }
        ]

        response = mock_llm.chat(messages)

        return {
            "answer": response.choices[0].message.content,
            "sources": list(dict.fromkeys(
                item["source"] for item in retrieved
            ))
        }


# Part 4: Evaluate retrieval and answers
def evaluate_rag(rag, overlap):
    evaluation_data = pd.read_csv(EVAL_FILE)

    total_answerable = 0
    expected_doc_found = 0
    expected_fact_found = 0

    total_unanswerable = 0
    fallback_correct = 0

    for _, row in evaluation_data.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"])
        expected_fact = str(row["expected_fact"])

        retrieved = rag.retrieve(question, k=3)
        result = rag.answer(question)
        answer = result["answer"]

        if expected_doc.lower() == "none":
            total_unanswerable += 1

            if answer.startswith("I could not find"):
                fallback_correct += 1

        else:
            total_answerable += 1

            retrieved_sources = [
                item["source"] for item in retrieved
            ]

            if expected_doc in retrieved_sources:
                expected_doc_found += 1

            if expected_fact in answer:
                expected_fact_found += 1

    print(f"\nEvaluation with overlap={overlap}")
    print("-" * 45)

    print(
        f"Expected document in top 3: "
        f"{expected_doc_found}/{total_answerable}"
    )

    print(
        f"Expected fact in answer: "
        f"{expected_fact_found}/{total_answerable}"
    )

    print(
        f"Correct fallback answers: "
        f"{fallback_correct}/{total_unanswerable}"
    )


# Main execution
if __name__ == "__main__":
    rag = PolicyRAG()

    print("POLICY RAG SYSTEM")
    print("=" * 45)

    print("Total policy files:", len(list(POLICY_DIR.glob("*.txt"))))
    print("Total chunks:", len(rag.chunks))

    print("\nChunks per policy:")

    for source in sorted(set(
        chunk["source"] for chunk in rag.chunks
    )):
        count = sum(
            1 for chunk in rag.chunks
            if chunk["source"] == source
        )

        print(f"{source}: {count}")

    print("\nTesting retrieval:")

    results = rag.retrieve("refund policy late delivery", k=3)

    for result in results:
        print(
            f"\nSource: {result['source']}"
            f"\nScore: {result['score']:.4f}"
            f"\nText: {result['text']}"
        )

    print("\nTesting answer generation:")

    test_result = rag.answer("What is the refund policy for late delivery?")

    print("Answer:", test_result["answer"])
    print("Sources:", test_result["sources"])
    evaluate_rag(rag, overlap=10)

    print("\nRunning zero-overlap evaluation...")

    zero_overlap_rag = PolicyRAG.__new__(PolicyRAG)
    zero_overlap_rag.chunks = load_policy_chunks(
        chunk_size=40,
        overlap=0
    )

    zero_overlap_rag.vectorizer = TfidfVectorizer()

    zero_overlap_rag.chunk_texts = [
        chunk["text"] for chunk in zero_overlap_rag.chunks
    ]

    zero_overlap_rag.chunk_vectors = (
        zero_overlap_rag.vectorizer.fit_transform(
            zero_overlap_rag.chunk_texts
        )
    )

    evaluate_rag(zero_overlap_rag, overlap=0)

