# Tweet Sentiment Analysis

A chatty bot for COVID-19 Related Tweet Sentiment Analysis

Current Structure:
```shell
.
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- sentiment
|   |-- __init__.py
|   |-- chatbot
|   |   |-- README.md
|   |   |-- __init__.py
|   |-- dataset
|   |   |-- README.md
|   |   |-- __init__.py
|   |   |-- create_dataset.py
|   |   |-- load_dataset.py
|   |-- logging_config.py
|   `-- model
|       |-- README.md
|       |-- __init__.py
|       `-- tf_idf_model.py
|-- setup.py
```

# Run Tweet Sentiment Analysis

To run tweet sentiment analysis locally, it assumed the user already have docker installed, steps below:

- `docker build -t tweet_sentiment_app -f sentiment/mlengine/Dockerfile`
- `docker run -d -p 9898:8501 tweet_sentiment_app`
- Visit the app at: `http://localhost:9898`

if not have docker installed, there's pre-setup service online:

visit: `https://johdev.asuscomm.com:9898` (it may not be always online 24 x 7)
