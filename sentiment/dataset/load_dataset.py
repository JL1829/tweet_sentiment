"""
Author: Lu ZhiPing
email: lu.zhiping@u.nus.edu
"""
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os

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
                 n_rows: int) -> None:
        self.client = MongoClient(
            MONGO_URL,
            username=PYMONGO_USERNAME,
            password=PYMONGO_PASSWORD
        )
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.max = n_rows
    
    # def __next__(self):
    #     cursor = self.collection.find()
    #     for item in cursor:
    #         if self.n <= self.max:
    #             return item
    #         else:
    #             raise StopIteration
    
    # def __iter__(self):
    #     self.n = 0
    #     return self

