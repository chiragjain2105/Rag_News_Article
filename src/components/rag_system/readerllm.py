from prompts.readerllm import RAG_PROMPT_TEMPLATE
from typing import List,Optional,Tuple
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS




def answer_with_rag(
        question:str,
        llm,
        knowledge_index,
        reranker=None,
        num_retrieved_docs:int=10,
        num_docs_final:int=3
)->Tuple[str,LangchainDocument]:
    
    relevant_docs=knowledge_index.similarity_search(question,k=num_retrieved_docs)
    relevant_docs=[doc.page_contet for doc in relevant_docs]

    if reranker:
        pass

    context="\nExtracted documents:\n"
    context+="".join([f"Document {str(i)}:::\n"+doc for i,doc in enumerate(relevant_docs)])

    final_prompt=RAG_PROMPT_TEMPLATE.format(context=context,question=question)

    answer=llm(final_prompt)

    return answer,relevant_docs
