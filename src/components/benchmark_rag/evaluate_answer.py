from langchain.prompts.chat import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from prompts.evaluation_prompt import EVALUATION_PROMPT
from langchain_groq import ChatGroq
import os
import json
from tqdm.auto import tqdm
# evaluate_prompt_template=ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content="You are a fair evaluator language model."),
#         HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT)
#     ]
# )

# eval_chat_model=ChatGroq(
#     temperature=0,
#     model="llama3-70b-8192"
# )

# evaluator_name="Groq_llama"

def evaluate_answers(
        answer_path:str,
        eval_chat_model,
        evaluator_name,
        evaluation_prompt_template:ChatPromptTemplate
)->None:
    answers=[]
    if os.path.isfile(answer_path):
        answers=json.load(open(answer_path,"r"))

        for experiment in tqdm(answers):
            if f"eval_score_{evaluator_name}" in experiment:
                continue
            eval_prompt=evaluate_prompt_template.format_messages(
                instruction=experiment["question"],
                response=experiment["generated_answer"],
                reference_answer=experiment["true_answer"]
            )

            eval_result=eval_chat_model(eval_prompt)
            feedback,score = [item.strip() for item in eval_result.content.split("[RESULT]")]
            experiment[f"eval_score_{evaluator_name}"]=score
            experiment[f"eval_feedback_{evaluator_name}"]=feedback
            

            with open(answer_path,"w") as f:
                json.dump(answers,f)
