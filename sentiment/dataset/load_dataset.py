"""
Author: Lu ZhiPing
email: lu.zhiping@u.nus.edu
"""
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os
from pprint import pformat

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")
PYMONGO_USERNAME = os.getenv("PYMONGO_USERNAME")
PYMONGO_PASSWORD = os.getenv("PYMONGO_PASSWORD")
MONGO_URL = os.getenv("MONGO_URL")


class LoadDataset:
    """
    Select the corresponding collection from MongoDB, iterate the document
    """

    def __init__(self, database_name: str, 
                 collection_name:str,
                 n_rows: int = 10000) -> None:
        self.client = MongoClient(
            MONGO_URL,
            username=PYMONGO_USERNAME,
            password=PYMONGO_PASSWORD
        )
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.max = n_rows
    
    def __len__(self) -> int:
        return self.collection.count_documents({})
    
    def __getitem__(self, index):
        return self.collection.find()[index]

    def __repr__(self) -> str:
        sample = self.collection.find()[0]
        return f"""
        Database: {str(self.db)},
        Collection: {str(self.collection)}
        Length : {self.collection.count_documents({})}
        Sample: {pformat(sample)}
        """

    def __iter__(self):
        for document in self.collection.find():
            yield document

    def to_pandas(self):
        """
        Iterate through the whole db collection, transform it into Pandas DataFrame and return
        """
        cursor = self.collection.find()
        return pd.DataFrame(list(cursor))
    
    def tokenize(self):
        raise NotImplementedError

    def update_one(self):
        raise NotImplementedError
    
    def update_many(self):
        raise NotImplementedError
    



if __name__ == "__main__":
    dataset = LoadDataset(
        database_name="PLP",
        collection_name="tweet"
    )
    print(dataset)
