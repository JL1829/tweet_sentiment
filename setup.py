from setuptools import setup, find_packages

setup(
    name="tweet_sentiment",
    version="0.0.1",
    description="COVID-19 Tweet Sentiment Analysis",
    python_requires=">=3.7",
    packages=find_packages(include=["tweet_sentiment", "tweet_sentiment.*"]),
    install_requires=[
        "pymongo==3.12.0",
        "numpy==1.23.1",
        "pandas=1.4.4"
    ]
)