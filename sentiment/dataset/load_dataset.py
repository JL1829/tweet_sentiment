"""
Author: Lu ZhiPing
email: lu.zhiping@u.nus.edu
"""
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os
import logging
from typing import Union, Callable
from tqdm import tqdm
from pprint import pformat
from random import randint

load_dotenv()
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_NAME = os.getenv("PROJECT_NAME")
PYMONGO_USERNAME = os.getenv("PYMONGO_USERNAME")
PYMONGO_PASSWORD = os.getenv("PYMONGO_PASSWORD")
MONGO_URL = os.getenv("MONGO_URL")


class LoadDataset:
    """
    Select the corresponding collection from MongoDB, iterate the document

    params:
    database_name: str, the name of the database want to connect
    collection_name: str, the name of the collection want to iterate
    n_rows: Union[int, str, float]: default value: 10000, How many rows want to return from database,
            Example:
                n_rows = 10000, it will only return first 10000 documents from MongoDB collection
                n_rows = "max", it will return all documents
                n_rows = 0.8, it will return 80% of the documents from MongoDB collection
    tokenizer: Callable, default value: None, A Callable to produce tokenization logic on input string.
            A Callable should take string as input, and List[str] as output.
    """

    def __init__(self, database_name: str,
                 collection_name: str,
                 n_rows: Union[int, str, float] = 10000,
                 tokenizer: Callable = None) -> None:
        self.tokenizer = tokenizer
        self.client = MongoClient(
            MONGO_URL,
            username=PYMONGO_USERNAME,
            password=PYMONGO_PASSWORD
        )
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.max_rows = self.collection.count_documents({})
        if isinstance(n_rows, int):
            if n_rows <= self.max_rows:
                self.n_rows = n_rows
            else:
                raise ValueError(
                    "n_row value cannot more than max collection value")
        elif isinstance(n_rows, str) and n_rows == "max":
            self.n_rows = self.max_rows
        elif isinstance(n_rows, float):
            if 0.1 <= n_rows <= 1.0:
                self.n_rows = self.max_rows * n_rows
            else:
                raise ValueError(
                    "Floating n_rows must within 0.1 -- 1.0 range")
        logger.info(
            f"Initialized Mongo Connection to db:{database_name}, collection: {collection_name}")

    def __len__(self) -> int:
        if self.n_rows <= self.max_rows:
            return self.n_rows
        else:
            raise ValueError(
                "Defined n_rows more than maximum documents in collection")

    def __getitem__(self, index):
        if index <= self.n_rows:
            return self.collection.find()[index]
        else:
            raise ValueError("Index out of range")

    def __repr__(self) -> str:
        index = randint(0, self.n_rows)
        sample = self.collection.find()[index]
        return f"""
        Database: {str(self.db)},
        Collection: {str(self.collection)}
        Length : {self.collection.count_documents({})}
        Sample: {pformat(sample)}
        """

    def __iter__(self):
        for document in self.collection.find().limit(self.n_rows):
            if not self.tokenizer:
                yield document
            else:
                if "Text" in document.keys():
                    document["token"] = self.tokenizer(document["Text"])
                    yield document
                elif "Tweet" in document.keys():
                    document["token"] = self.tokenizer(document["Tweet"])
                    yield document
                else:
                    raise KeyError("Text or Tweet can't find in document")

    def to_pandas(self):
        """
        Iterate through the whole db collection, transform it into Pandas DataFrame and return
        """
        logger.info(
            f"Returning Pandas DataFrame with maximum row: {self.n_rows}")
        cursor = self.collection.find().limit(limit=self.n_rows)
        documents = [doc for doc in tqdm(cursor, total=self.n_rows)]
        return pd.DataFrame(documents)

    def tokenize(self):
        raise NotImplementedError

    def update_one(self):
        raise NotImplementedError

    def update_many(self):
        raise NotImplementedError


if __name__ == "__main__":
    dataset = LoadDataset(
        database_name="PLP",
        collection_name="AStarCOVID",
        n_rows=100
    )
    print(dataset)
    print(len(dataset))
