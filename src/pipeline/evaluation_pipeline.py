from langchain.prompts.chat import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from prompts.evaluation_prompt import EVALUATION_PROMPT
from langchain_groq import ChatGroq
import os
import json
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from src.logger.logging import logging
from src.components.rag_system.retriever_embedding import load_embeddings
from langchain.docstore.document import Document as LangchainDocument
from src.components.benchmark_rag.run_rag_test import run_rag_tests
from src.components.benchmark_rag.evaluate_answer import evaluate_answers
from src.components.data_preparation import eval_dataset
import glob
import pandas as pd


load_dotenv()

READER_LLM=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False
)

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc['articleBody'],metdata={"source":doc['source'],"title":doc['title']}) for doc in random_articles
]


evaluate_prompt_template=ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT)
    ]
)

eval_chat_model=ChatGroq(
    temperature=0,
    model="llama3-70b-8192"
)

evaluator_name="Groq_llama"

if not os.path.isdir("./output"):
    os.mkdir("./output")

for chunk_size in [2500]:
    for embeddings in ["embed-english-light-v3.0"]:
        for rerank in [False]:
            settings_name=f"chunk_size:{chunk_size}_embedding:{embeddings.replace('/','~')}_rerank:{rerank}_readerLLM:{READER_LLM}"
            output_file_name=f"./output/rag_{settings_name}"

            logging.info(f"Running RAG with setting: {settings_name}")

            logging.info("Loading knowledge base embedding..........")

            knowledge_index=load_embeddings(
                RAW_KNOWLEDGE_BASE,
                chunk_size=chunk_size,
                embeddings_model=embeddings
            )

            logging.info("Running RAG...........")
            reranker=None

            run_rag_tests(
                eval_dataset=eval_dataset,
                llm=READER_LLM,
                knowledge_index=knowledge_index,
                output_file=output_file_name,
                reranker=reranker,
                verbose=False,
                test_settings=settings_name
            )


            logging.info("Evaluating RAG...........")

            evaluate_answers(
                output_file_name,
                eval_chat_model,
                evaluator_name,
                evaluate_prompt_template
            )


outputs=[]
for file in glob.glob("./output/*.json"):
    output=pd.DataFrame(json.load(open(file,"r")))
    output["settings"]=file
    outputs.append(output)
result=pd.concat(outputs)

result["eval_score_Groq_llama"]=result["eval_score_Groq_llama"].apply(lambda x: int(x) if isinstance(x,str) else 1)
result["eval_score_Groq_llama"]=(result["eval_score_Groq_llama"]-1)/4 

average_scores=result.groupby("settings")["eval_score_Groq_llama"].mean()

logging.info(f"Final result score is {average_scores}")
