#News Article RAG system along with its evaluation using Open source Models


This repository is dedicated to developing a RAG (Retrieval-Augmented Generation) system and evaluating its performance 
using a synthetic dataset and an LLM-as-a-judge framework to measure accuracy.

RAG Evaluation Pipeline:
Synthetic Dataset Generation: Create question-answer pairs from the ingested dataset. Approximately 300 questions were 
generated from around 2500 news articles using Mistral's LLM.
Relevance Computation: Utilize LLM-as-a-judge to score the questions generated in the previous step based on criteria 
outlined in relevant literature.

Evaluation Criteria:
Groundedness: Can the question be answered based on the given context?
Standalone: Is the question understandable without additional context?

All questions were scored based on these criteria and filtered out those questions having both scores greater then or equal to 4.

Development of RAG System

Pipeline for RAG Development (currently focusing on a NAIVE RAG approach):
Data Ingestion: Retrieve the dataset and filter news articles related to the Israel-Hamas conflict.
Data Indexing: Use COHERE Embedding to embed the segmented data with RecursiveTextSplitter and index it in the FAISS vectorstore.
Retrieval: Extract similar documents to the query from the vectorstore using COSINE similarity.
Generation: Feed the retrieved documents and the query to the LLM, specifically using MISTRAL 7b.

Benchmarking RAG

Benchmarking allows us to assess how closely the model's output aligns with human expectations. To achieve this, 
we employed LLAMA as a judge to evaluate the generated answers, scoring them on a scale of 1 to 5. The answers to the questions
in evaluation dataset serves as a reference for scoring the generated outputs.

Here in our case we obtained a normalized accuracy score of 0.6994