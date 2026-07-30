from ai_orchestrator.retrievers.base import BaseRetriever
from ai_orchestrator.retrievers.tfidf_retriever import TfidfRetriever
from ai_orchestrator.retrievers.bm25_retriever import BM25Retriever

__all__ = [
    "BaseRetriever",
    "TfidfRetriever",
    "BM25Retriever",
    # Heavy-dependency retrievers are imported lazily inside execution_service
    # to avoid hard-failing when sentence-transformers / faiss / chromadb are
    # not installed.  Import them directly if you need them:
    #   from ai_orchestrator.retrievers.sentence_transformer_retriever import SentenceTransformerRetriever
    #   from ai_orchestrator.retrievers.faiss_retriever import FaissRetriever
    #   from ai_orchestrator.retrievers.chroma_retriever import ChromaRetriever
]
