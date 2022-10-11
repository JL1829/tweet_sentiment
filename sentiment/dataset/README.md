# Dataset loading package

This subpackage intend to make connection to MongoDB datset service related code

How to use: 

```py
from sentiment.dataset.load_dataset import LoadDataset


dataset = LoadDataset(
    database_name="PLP",
    collection_name="tweet"
)

# __repr__ method
print(dataset)
----------------
Database: Database(MongoClient(host=['192.168.50.72:27017'], document_class=dict, tz_aware=False, connect=True), 'PLP'),
        Collection: Collection(Database(MongoClient(host=['192.168.50.72:27017'], document_class=dict, tz_aware=False, connect=True), 'PLP'), 'tweet')
        Length : 8507
        Sample: {'Date': '2021-11-29 17:09:07+00:00',
 'Tweet': 'SINGAPORE - A new Covid-19 vaccine could be available in Singapore, '
          'with American firm Novavax having applied for interim authorisation '
          'by the Health Sciences Authority (HSA). '
          'straitstimes.com/singapore/nova… via @stcom #CircuitBreaker '
          '#DORSCON',
 'Tweet_Origin': 'SINGAPORE - A new Covid-19 vaccine could be available in '
                 'Singapore, with American firm Novavax having applied for '
                 'interim authorisation by the Health Sciences Authority '
                 '(HSA). https://t.co/LTsUa6WpHm via @stcom #CircuitBreaker '
                 '#DORSCON',
 'User': '24x7Page',
 '_id': ObjectId('6343f3b26314f4b38b412561'),
 'hashtag': "['CircuitBreaker', 'DORSCON']",
 'huggingFace_label': 'NEGATIVE',
 'lang': 'en',
 'url': 'https://twitter.com/24x7Page/status/1465367231718641665'}

# __len__ method
len(dataset)
>>> 8507

# __getitem__ method
dataset[100]
--------------
{'_id': ObjectId('6343f3b36314f4b38b4125c5'),
 'Date': '2021-07-22 20:41:20+00:00',
 'url': 'https://twitter.com/masterdougles/status/1418310214768492544',
 'User': 'masterdougles',
 'lang': 'en',
 'Tweet': 'Covid19 Singapore Predictions using QMDJ Prediction youtu.be/mJNqMcTsfC8 #qimen #metaphysics #covid19 #singapore #circuitbreaker',
 'Tweet_Origin': 'Covid19 Singapore Predictions using QMDJ Prediction https://t.co/fDmtCQbgVC #qimen #metaphysics #covid19 #singapore #circuitbreaker',
 'hashtag': "['qimen', 'metaphysics', 'covid19', 'singapore', 'circuitbreaker']",
 'huggingFace_label': 'NEGATIVE'}

# convert to pandas: 
df = dataset.to_pandas()

# iteration, __iter__ method:
from tqdm import tqdm

for item in tqdm(dataset, total=len(dataset)):
    print(item)
```
