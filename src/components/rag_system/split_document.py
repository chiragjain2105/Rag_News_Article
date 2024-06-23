from typing import Optional,List,Tuple
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.components.data_preparation import random_articles



def split_documents(
        chunk_size:int,
        knowledge_base: List[LangchainDocument]
):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,chunk_overlap=int(chunk_size/10),add_start_index=True,separators=["\n","\n\n",".",""," "]
    )

    docs_processed=[]

    for doc in knowledge_base:
        docs_processed+=text_splitter.split_documents([doc])
    
    unique_texts={}
    docs_processed_unique=[]

    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content]=True
            docs_processed_unique.append(doc)
    
    return docs_processed_unique
