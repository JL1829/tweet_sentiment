from pymongo import MongoClient
import numpy as np
import pandas
from transformers import AutoTokenizer, AutoModel, pipeline,AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-uncased"
#model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)


mg_client = MongoClient("mongodb://localhost:27017/")
mg_db = mg_client["PLP"]
mg_col = mg_db["COVID_FAQ"]


excel_data_df = pandas.read_excel('COVID19_FAQ.xlsx', sheet_name='Overrall')
#"""
for i, row in excel_data_df.iterrows():
    features = fe(row["Question"].strip())
    covid_dict = { "_id": i, "title": row["Title"].strip(),"question":row["Question"].strip(),"question_features":features,"answer":row["Answer"].strip()}
    result = mg_col.insert_one(covid_dict)
    print("inserted id is:{0}".format(result.inserted_id))
    
#"""
