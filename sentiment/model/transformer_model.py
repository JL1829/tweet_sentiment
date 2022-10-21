"""
Author: Lu.ZhiPing
Email: lu.zhiping@u.nus.edu
"""
from sentiment.dataset.load_dataset import LoadDataset
from sentiment.dataset.tokenizer import SimpleTokenizer
from argparse import ArgumentParser
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate


def prepare_dataset(dataset, column_names):
    df = dataset.to_pandas()
    df = df[column_names].copy()
    return df


def dataset_convert(df):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(doc):
        return tokenizer(doc['Text'])

    le = LabelEncoder()
    df["labels"] = le.fit_transform(df[args.label_column])
    label2id = dict()
    for index, element in enumerate(le.classes_):
        label2id[element] = index
    id2label = {value:key for key, value in label2id.items()}
    train, test = train_test_split(df, test_size=0.1)
    hg_dataset_train = Dataset.from_pandas(train)
    hg_dataset_test = Dataset.from_pandas(test)
    hg_dataset_train = hg_dataset_train.train_test_split(test_size=0.2)

    tokenized_dataset_train = hg_dataset_train.map(tokenize_function, batched=True)
    tokenized_dataset_test = hg_dataset_test.map(tokenize_function, batched=True)
    tokenized_dataset_train = tokenized_dataset_train.remove_columns(["sentiment_category", "Text", "tokens", "__index_level_0__"])
    tokenized_dataset_test = tokenized_dataset_test.remove_columns(["sentiment_category", "Text", "tokens", "__index_level_0__"])
    tokenized_dataset_train.set_format("torch")
    tokenized_dataset_test.set_format("torch")

    return tokenized_dataset_train, tokenized_dataset_test, tokenizer, le, id2label


def train_func(train_dataloader, le, id2label, epoch):
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le.classes_),
                                                               id2label=id2label)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = epoch
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    return model

def eval_func(test_loader, model):
    metric = evaluate.load("accuracy")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        y_pred.append(predictions.detach().cpu().numpy())
        y_true.append(batch["labels"].detach().cpu().numpy())

    accuracy = metric.compute()
    return y_pred, y_true, accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("col_name", type=str, help="The column that need to keep for tokenization")
    parser.add_argument("label_column", type=str, help="The label column name")
    parser.add_argument("epoch", type=int, help="How many epoch the model should train")

    args = parser.parse_args()
    simple_tokenizer = SimpleTokenizer()
    dataset = LoadDataset(
        database_name="PLP",
        collection_name="AStarCOVID",
        n_rows="max",
        tokenizer=simple_tokenizer,
        column_name="Text"
    )
    df = prepare_dataset(dataset, column_names=args.col_name)
    tokenized_dataset_train, tokenized_dataset_test, tokenizer, le, id2label = dataset_convert(df)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_dataset_train["train"], shuffle=True, batch_size=16, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_dataset_train["test"], batch_size=16, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_dataset_test, batch_size=16, collate_fn=data_collator
    )

    model = train_func(train_dataloader, le, id2label, args.epoch)
    y_pred, y_true, accuracy = eval_func(test_dataloader, model)
