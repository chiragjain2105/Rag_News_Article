from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_cohere import CohereEmbeddings
from langchain.docstore.document import Document as LangchainDocument
import os
from dotenv import load_dotenv
from typing import List,Optional,Tuple
from src.components.rag_system.split_document import split_documents
from src.logger.logging import logging


load_dotenv()

def load_embeddings(
        langchain_docs:List[LangchainDocument],
        chunk_size:int,
        embeddings_model
)->FAISS:
    
    logging.info("Accessing embedding")
    embedding_model=CohereEmbeddings(model=embeddings_model)

    logging.info("saving or importing vector index")
    index_name=f"index_chunk_{chunk_size}_embeddings_{embeddings_model.replace('/', '_')}"
    index_folder_path=f"./data/indexes/{index_name}"
    
    if os.path.isdir(index_folder_path):
        logging.info("imported index")
        return FAISS.load_local(
            index_folder_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE
        )
    else:
        logging.info("saving index")
        docs_processed=split_documents(
            chunk_size,
            langchain_docs
        )

        knowledge_index=FAISS.from_documents(
            docs_processed,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE
        )
        knowledge_index.save_local(index_folder_path)
        return knowledge_index
