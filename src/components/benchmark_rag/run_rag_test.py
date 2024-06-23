import datasets
from typing import Optional,List,Tuple
from tqdm.auto import tqdm
import json
from src.components.rag_system.readerllm import answer_with_rag

def run_rag_tests(
        eval_dataset:datasets.Dataset,
        llm,
        knowledge_index,
        output_file:str,
        reranker:None,
        verbose:Optional[bool]=True,
        test_settings:Optional[str]=None
):
    try:
        with open(output_file,"r") as f:
            outputs=json.load(f)
    except:
        outputs=[]

    for example in tqdm(eval_dataset):
        question=example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer,relevant_docs=answer_with_rag(
            question,
            llm,
            knowledge_index,
            reranker=reranker
        )

        if verbose:
            print("-------------------------------------------------------------------------------")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"True Answer: {example["answer"]}")
        

        result = {
            "question":question,
            "true_answer":example["answer"],
            "source_doc":example["source"],
            "generated_answer":answer,
            "retrieved_docs":[doc for doc in relevant_docs]
        }

        if test_settings:
            result["test_settings"]=test_settings
        
        outputs.append(result)

        with open(output_file,"w") as f:
            json.dump(outputs,f)