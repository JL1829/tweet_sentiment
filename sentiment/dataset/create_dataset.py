"""
Author: Lu ZhiPing
email: lu.zhiping@u.nus.edu
"""
from dotenv import load_dotenv
import os
from tqdm import tqdm
from time import sleep
from typing import List
import pandas as pd
import logging
from argparse import ArgumentParser
from pymongo import MongoClient
import tweepy
load_dotenv()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TWITTER_BEARE_TOKEN")
PROJECT_NAME = os.getenv("PROJECT_NAME")
PYMONGO_USERNAME = os.getenv("PYMONGO_USERNAME")
PYMONGO_PASSWORD = os.getenv("PYMONGO_PASSWORD")
MONGO_URL = os.getenv("MONGO_URL")


def chunks(lst: List, n: int) -> List[List]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_batch_tweets(ids: List[List], client) -> pd.DataFrame:
    """
    Get a batch of Tweet ID and GET from Twitter APIv2 and convert into Pandas DataFrame
    """
    tweets = []
    logger.info("Start Pulling Tweets from Twitter APIv2")
    for idx, item in tqdm(enumerate(ids), total=len(ids)):
        if idx == 0 or idx % 298 != 0:
            responds = client.get_tweets(ids=item)
            if responds.data:
                for doc in responds.data:
                    tweets.append([doc.id, doc.text])
        elif idx % 298 == 0:
            logger.info("Now sleep for 15 minutes")
            sleep(900)

    return pd.DataFrame(tweets)


def insert_db(dataframe: pd.DataFrame, database_name: str, collection_name: str) -> None:
    mg_client = MongoClient(
        MONGO_URL,
        username=PYMONGO_USERNAME,
        password=PYMONGO_PASSWORD
    )

    db = mg_client[database_name]
    collection = db[collection_name]
    logger.info("Now start to insert document to db")
    for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        document = dict()
        for i in row.index:
            document[i] = row[i]
        collection.insert_one(document)


if __name__ == "__main__":
    tw_client = tweepy.Client(bearer_token=TOKEN)
    parser = ArgumentParser()
    parser.add_argument("ids_path", type=str, help="the path to the file contains tweets id")
    parser.add_argument("db_name", type=str, help="MongoDB's database name")
    parser.add_argument("collection_name", type=str, help="MongoDB Database Collection Name")
    args = parser.parse_args()

    if args.ids_path.endswith(".csv"):
        df = pd.read_csv(args.ids_path)
        tweet_ids = df.tweet_ID.to_list()
    elif args.ids_path.endswith(".txt"):
        with open(args.ids_path) as file:
            tweet_ids = file.read().splitlines()

    process_list = chunks(tweet_ids, 100)
    df = get_batch_tweets(process_list, tw_client)
    insert_db(df, args.db_name, args.collection_name)
    logger.info("Add new documents to db done.")
