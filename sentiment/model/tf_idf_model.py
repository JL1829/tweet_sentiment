"""
Author: Lu ZhiPing
email: lu.zhiping@u.nus.edu

Run a bag-of-word approach TF-IDF model for selected dataset, it does:
* Select Vocabulary for default hyperparameter
* Select Vocabulary for specific hyperparameter
* Tokenize with default Regex Tokenize pattern
* Iterate through the dataset and create vectorized matrix
* run model training experiment
* save artifact (local/cloud), such as vocabulary, vectorizer, model, and metric
"""
from sentiment.dataset.load_dataset import LoadDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from helper import NpEncoder
import numpy as np
import json
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import lightgbm as lgb


def get_matrix_vocab(dataset, column_name):
    vectorizer = TfidfVectorizer(
        max_features=2000
    )

    def yield_from_mongo(ds):
        for item in ds:
            yield item[column_name]
    gen = yield_from_mongo(ds=dataset)
    matrix = vectorizer.fit_transform(gen)
    vocab = vectorizer.vocabulary_
    return matrix, vocab


if __name__ == "__main__":
    dataset = LoadDataset(
        database_name="PLP",
        collection_name="AStarCOVID",
        n_rows=10000,
        tokenizer=None
    )
    le = LabelEncoder()
    labels = [item["sentiment_category"] for item in dataset]
    labels = le.fit_transform(labels)
    matrix, vocab = get_matrix_vocab(dataset, "Text")
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, labels, test_size=0.2, random_state=42
    )

    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'num_iterations': 500,
        "early_stopping_rounds": 25
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(
        lgbm_params,
        lgb_train,
        valid_sets=lgb_eval,
        verbose_eval=100
    )
    y_pred = gbm.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    ACC = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="micro")
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    CN = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # save experiment result
    with open("experiment_result.txt", "w") as file:
        file.write(f"Accuracy: {ACC}\n")
        file.write(f"F1: {F1}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"Confusion Matrix: \n{CN}\n")
        file.write(f"Classification Report: \n{report}")

    # save vocabulary
    with open("vocabulary.json", "w") as file:
        json.dump(vocab, file, cls=NpEncoder)
