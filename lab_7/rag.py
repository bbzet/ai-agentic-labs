import glob
import json
import os
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE, "kb_alatech")
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5


def load_kb():
    meta = {m["doc_id"]: m for m in json.load(open(os.path.join(KB_DIR, "metadata.json"), encoding="utf-8"))}
    docs = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "docs", "*.md"))):
        doc_id = os.path.basename(path)[:-3]
        docs.append({"doc_id": doc_id, "text": open(path, encoding="utf-8").read(), **meta[doc_id]})
    questions = json.load(open(os.path.join(KB_DIR, "questions.json"), encoding="utf-8"))
    return docs, questions


class Pipeline:
    def __init__(self, docs, questions):
        self.docs = docs
        self.questions = questions
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
        self.client = chromadb.EphemeralClient()
        self.collections = {}

    def make_chunks(self, chunk_size, overlap):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = []
        for d in self.docs:
            for i, piece in enumerate(splitter.split_text(d["text"])):
                chunks.append({"id": f"{d['doc_id']}#{i}", "text": piece, "doc_id": d["doc_id"]})
        return chunks

    def build_index(self, chunks, name):
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        col = self.client.create_collection(name, embedding_function=self.ef, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), 256):
            b = chunks[i:i + 256]
            col.add(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                    metadatas=[{"doc_id": c["doc_id"]} for c in b])
        return col

    def add_config(self, chunk_size, overlap):
        chunks = self.make_chunks(chunk_size, overlap)
        col = self.build_index(chunks, f"kb_{chunk_size}_{overlap}")
        self.collections[(chunk_size, overlap)] = col
        return chunks

    def retrieve(self, config, question, k=TOP_K):
        r = self.collections[config].query(query_texts=[question], n_results=k,
                                           include=["documents", "metadatas", "distances"])
        return [{"text": t, "doc_id": m["doc_id"], "distance": d}
                for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def ranks(self, config, q):
        found = self.retrieve(config, q["question"])
        return [i for i, c in enumerate(found) if is_relevant(c["text"], q)]

    def evaluate(self, config):
        hit1 = hit5 = mrr = 0
        for q in self.questions:
            ranks = self.ranks(config, q)
            if ranks:
                hit5 += 1
                hit1 += ranks[0] == 0
                mrr += 1 / (ranks[0] + 1)
        n = len(self.questions)
        return {"hit@1": hit1 / n, "hit@5": hit5 / n, "MRR": round(mrr / n, 3)}


def is_relevant(text, q):
    return any(s in text for s in q["gold_snippets"])


def kb_stats(docs, questions):
    print("Документов:", len(docs), "| вопросов:", len(questions))
    print("По типам:", dict(Counter(d["type"] for d in docs)))
    for t in ["policy", "ticket", "order", "faq"]:
        lens = [len(d["text"]) for d in docs if d["type"] == t]
        print(f"  {t:7s} средняя длина: {sum(lens) // len(lens)} символов")
    print("\nТестовые вопросы:")
    for q in questions:
        print(f"  {q['id']}: {q['question']}")
