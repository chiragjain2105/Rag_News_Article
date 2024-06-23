from src.components.data_ingestion import qa_data,news_article,random_articles
import re
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain.docstore.document import Document as LangchainDocument
import datasets
from tqdm.auto import tqdm
from prompts.data_prep_prompts import QA_generation_prompt,question_groundedness_critique_prompt,question_standalone_critique_prompt
from src.logger.logging import logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def data_preparation():
    eval_dataset = datasets.Dataset.from_pandas(qa_data,split="train",preserve_index=False)
    return qa_data,random_articles,eval_dataset


# keywords=['israel','hamas','gaza']

# def clean_text(text):
#     text=re.sub(r'\W+'," ",text)
#     text=text.lower()
#     return text

# def is_relevant(text):
#     return any(keyword in text for keyword in keywords)

# filtered_articles,y_train,news=[],[],[]

# for item in news_article:
#     article_body=clean_text(item['articleBody'])
#     article_title=clean_text(item['title'])
#     news.append(article_title)
#     if is_relevant(article_body) or is_relevant(article_title):
#         filtered_articles.append(item)
#         y_train.append(1)
#     else:
#         y_train.append(0)


# random_articles = random.sample(filtered_articles,2500)
# ----------------------------------------------------------------------------------
random_articles=random_articles
# ----------------------------------------------------------------------------------

# langchain_docs = [
#     LangchainDocument(page_content=doc['articleBody'],metadata={"source":doc['source'],"title":doc['title']}) for doc in random_articles
#     ]

# text_splitter=RecursiveCharacterTextSplitter(chunk_size=2500,chunk_overlap=200,add_start_index=True,separators=["\n","\n\n",".",""," "])

# docs_processed=[]
# for doc in langchain_docs:
#     docs_processed+=text_splitter.split_documents([doc])


# llm=HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     task="task-generation",
#     max_new_tokens=1000,
#     do_sample=False
# )

# logging.info("importing QA_generation_prompt prompt...............")

# def generate_response(llm,prompt):
#     return llm(prompt)

# N_GENERATIONS=100
# logging.info(f"Generating {N_GENERATIONS} factoid questions...")

# outputs=[]
# for sampled_context in tqdm(random.sample(docs_processed,N_GENERATIONS)):
#     prompt=QA_generation_prompt.format(context=sampled_context.page_content)
#     response=llm(prompt)

#     question=response.split("Factoid question:")[1].split("Answer:")[0].strip()
#     answer=response.split("Answer:")[1].strip()

#     outputs.append({
#         "question":question,
#         "answer":answer,
#         "context":sampled_context.page_content,
#         "source":sampled_context.metadata["source"],
#         "title":sampled_context.metadata["title"]
#     })


logging.info("Generating critique for each QA couple")

# for output in tqdm(outputs):
#     evaluations = {
#         "question_groundedness":generate_response(
#             llm,question_groundedness_critique_prompt.format(context=output["context"],question=output["question"])
#         ),
#         "question_standalone":generate_response(
#             llm,question_standalone_critique_prompt.format(question=output["question"])
#         )
#     }
#     try:
#         for criterion,evaluation in evaluations.items():
#             score,eval=(
#                 int(evaluation.split("Total rating: ")[-1].strip()),
#                 evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1]
#             )
#             output.update(
#                 {
#                     f"{criterion}_score":score,
#                     f"{criterion}_evaluation":eval
#                 }
#             )
#     except Exception as e:
#         continue


# logging.info("Generating dataframe....")
# df_outputs=pd.DataFrame(outputs)

# logging.info("filtering dataframe.....")
# filtered_df_outputs = df_outputs.loc[
#     (df_outputs["question_groundedness_score"]>=4)
#     & (df_outputs["question_standalone_score"]>=4)
# ]
# qa_data=filtered_df_outputs

logging.info("eval_dataset prepared....")
eval_dataset = datasets.Dataset.from_pandas(qa_data,split="train",preserve_index=False)