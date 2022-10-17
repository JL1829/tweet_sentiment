from setuptools import setup, find_packages

setup(
    name="sentiment",
    version="0.0.1",
    description="COVID-19 Tweet Sentiment Analysis",
    python_requires=">=3.8",
    packages=find_packages(include=["sentiment", "sentiment.*"]),
    install_requires=[
        "pymongo",
        "numpy",
        "pandas",
        "snscrape",
        "python-dotenv",
        "tweepy",
        "tqdm",
        "scikit-learn",
        "lightgbm"
    ]
)
