import os
from typing import Any, List, Optional, cast

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.models.core import LDU

load_dotenv()


class VectorStore:
    """
    Manages the FAISS vector store for semantic LDU retrieval.
    """

    def __init__(self, index_name: str = "main_index"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.index_name = index_name
        self.persist_dir = ".refinery/vectorstore"

        # OpenRouter-compatible embedding setup
        from pydantic import SecretStr

        self.embeddings = OpenAIEmbeddings(
            api_key=SecretStr(self.api_key) if self.api_key else None,
            base_url="https://openrouter.ai/api/v1",
            model="text-embedding-3-small",
        )
        self.vector_db: Optional[FAISS] = None

    def _ldus_to_documents(self, ldus: List[LDU]) -> List[Document]:
        """Converts internal LDU model to LangChain Document."""
        docs = []
        for ldu in ldus:
            doc = Document(
                page_content=ldu.content,
                metadata={
                    "doc_id": ldu.doc_id,
                    "chunk_id": ldu.chunk_id,
                    "chunk_type": ldu.chunk_type,
                    "page_refs": ldu.page_refs,
                    "parent_section": ldu.parent_section,
                    "content_hash": ldu.content_hash,
                    "bbox": ldu.bounding_box.model_dump() if ldu.bounding_box else None,
                },
            )
            docs.append(doc)
        return docs

    def ingest_ldus(self, ldus: List[LDU]) -> None:
        """Embeds and adds LDUs to the vector store."""
        if not ldus:
            return

        lc_docs = self._ldus_to_documents(ldus)

        if self.vector_db is None:
            # Check if index exists on disk
            try:
                self.vector_db = FAISS.load_local(
                    self.persist_dir, self.embeddings, allow_dangerous_deserialization=True
                )
                self.vector_db.add_documents(lc_docs)
            except Exception:
                # Create new
                self.vector_db = FAISS.from_documents(lc_docs, self.embeddings)
        else:
            self.vector_db.add_documents(lc_docs)

        # Persist
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vector_db.save_local(self.persist_dir)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Performs a semantic search."""
        if self.vector_db is None:
            try:
                self.vector_db = FAISS.load_local(
                    self.persist_dir, self.embeddings, allow_dangerous_deserialization=True
                )
            except Exception:
                return []

        return cast(List[Document], self.vector_db.similarity_search(query, k=k))
