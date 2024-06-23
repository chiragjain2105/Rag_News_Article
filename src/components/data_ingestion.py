import json
import pandas as pd

with open("data/random_articles.json",'r',encoding='utf-8') as f:
    random_articles=json.load(f)

with open("data/news.article.json",'r',encoding='utf-8') as f:
    news_article=json.load(f)

qa_data = pd.read_csv("data\qa_data.csv")
qa_data.drop(columns=['Unnamed: 0'],inplace=True)
# print(qa_data.shape)