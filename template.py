import os
from pathlib import Path

list_of_files = [
    ".github/workflows/.gitkeep",
    ".env",
    "prompts/__init__.py",
    "prompts/data_prep_prompts.py",
    "prompts/readerllm.py",
    "prompts/evaluation_prompt.py",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_preparation.py",
    "src/components/rag_system/split_document.py",
    "src/components/rag_system/retriever_embedding.py",
    "src/components/rag_system/readerllm.py",
    "src/components/benchmark_rag/run_rag_test.py",
    "src/components/benchmark_rag/evaluate_answer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/evaluation_pipeline.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "src/logger/logging.py",
    "src/exception/exception.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/experiment.ipynb"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass