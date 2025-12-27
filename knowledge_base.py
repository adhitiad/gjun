import json
import logging
import time
from typing import cast

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from config import settings

logger = logging.getLogger("KnowledgeBase")


class MarketMemory:
    def __init__(self):
        self.pc = None
        self.index = None
        # Menggunakan model embedding yang ringan tapi akurat
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def init_pinecone(self):
        if not settings.PINECONE_API_KEY:
            logger.warning("⚠️ Pinecone API Key missing. Memory disabled.")
            return
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index_name = settings.PINECONE_INDEX

            # Cek apakah index ada, jika tidak buat baru
            existing_indexes = [i.name for i in self.pc.list_indexes()]
            if index_name not in existing_indexes:
                self.pc.create_index(
                    name=index_name,
                    dimension=384,  # Sesuai model all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            self.index = self.pc.Index(index_name)
            logger.info("✅ Pinecone Connected (Long-term Memory Active)")
        except Exception as e:
            logger.error(f"Pinecone Init Error: {e}")

    def store_experience(self, text: str, metadata: dict):
        """Menyimpan pengalaman (Trade result, News summary)"""
        if self.index:
            try:
                vec = self.embedder.embed_query(text)
                # ID unik berdasarkan timestamp
                uid = f"exp_{int(time.time())}"
                self.index.upsert(vectors=[(uid, vec, {"text": text, **metadata})])
            except Exception as e:
                logger.error(f"Store Memory Error: {e}")

    def recall_similar_situations(self, query: str, top_k=3):
        """Mengingat situasi serupa di masa lalu (RAG)"""
        if self.index:
            try:
                vec = self.embedder.embed_query(query)
                res = self.index.query(vector=vec, top_k=top_k, include_metadata=True)

                # Format hasil untuk LLM
                memories = []
                for m in res.matches:
                    score = m.score
                    text = m.metadata.get("text", "")
                    memories.append(f"- (Similiarity: {score:.2f}) {text}")

                return "\n".join(memories)
            except Exception as e:
                logger.error(f"Recall Error: {e}")
        return "No memory available."


memory_bank = MarketMemory()
