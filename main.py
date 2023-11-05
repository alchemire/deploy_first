from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
import pandas as pd
import pickle
import csv
import transformers
from transformers import BertModel, BertJapaneseTokenizer


app = FastAPI()

# define input data type
class text(BaseModel):
    input_text: str


# enbeddingしたリストを読み込み。
load_embeddings = []
# with open(r"C:\Users\wanna\OneDrive\デスクトップ\deploy_first\embedding_list.pkl", "rb") as file:
with open("embedding_list.pkl", "rb") as file:
  loaded_tensor_list = pickle.load(file)
  for tensor in loaded_tensor_list:
     load_embeddings.append(tensor)

# テキストと作者の読み込み
text_list=[]
author_list=[]
# with open(r"C:\Users\wanna\OneDrive\デスクトップ\deploy_first\under_1000.csv", mode="r", encoding="utf-8") as file:
with open("under_1000.csv", mode="r", encoding="utf-8") as file:
   csv_reader = csv.reader(file)
   next(csv_reader)
   for row in csv_reader:
      text_list.append(row[1])
      author_list.append(row[0])

# use cpu to smaller one >> add .to(device) bert_model and input_text 
device = torch.device('cpu')

# モデルの読み込み
model_name='cl-tohoku/bert-base-japanese-whole-word-masking'
bert_tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name).to(device)
cos =torch.nn.CosineSimilarity(dim=0)

@app.get("/")
def index():
    return{"温故知新": "類似度の高い青空文庫の作品を返す"}

# embedding input_text, then sort by similarity 
@app.post("/embedding")
def get_bert_embeddings_similar_sort(input_text: text):    
    bert_model.eval()
    tokenized_sent = bert_tokenizer.encode(
        input_text.input_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)
    attention_mask=[int(token_id) > 0 for token_id in tokenized_sent[0]]
    
    with torch.no_grad():
        result = bert_model(tokenized_sent, attention_mask=torch.tensor(attention_mask).unsqueeze(0))
        text_emb = result[1][0]

    # cos 類似度の算出
    similar_list = []
    for sentence_id, sentence_vector in enumerate(load_embeddings):
        similar_list.append([sentence_id, cos(text_emb, sentence_vector)])
    df_similars_list = pd.DataFrame(similar_list, columns=["sentence_id", "cos_similarity"])

    # 類似度の高い順に降順で並び替え、top5を返す
    sorted_df_similarity = df_similars_list.sort_values("cos_similarity", ascending=False)
    # top_similar = sorted_df_similarity.iloc[0]
    top5_similar = sorted_df_similarity[:5]
    # top_id = top_similar["sentence_id"]
    top5_id_list = [id for id in top5_similar["sentence_id"]]
    #response_data={
     #  "similarity_score" : round(float(top_similar["cos_similarity"]), 4),
      # "top_text" : text_list[top_id],
       #"top_author" : author_list[top_id]
    #}
    response_data={
        "similarity_score" : [round(float(similarity), 4) for similarity in top5_similar["cos_similarity"]],
        "top5_text" : [text_list[i] for i in top5_id_list],
        "top5_author" : [author_list[i] for i in top5_id_list]
    }
    
    return response_data