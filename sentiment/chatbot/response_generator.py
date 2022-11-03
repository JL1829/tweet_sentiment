import sys
from pymongo import MongoClient
import numpy as np
import pandas
import torch
from scipy.spatial.distance import cosine
import logging
from transformers import AutoTokenizer, AutoModel, pipeline,AutoModelForQuestionAnswering



max_length = 512 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
n_best_size = 30
max_answer_length = 512

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
saved_model = AutoModelForQuestionAnswering.from_pretrained("covid_model/distilbert-base-uncased-covid",use_auth_token="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")
model = AutoModel.from_pretrained(model_checkpoint)
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)


mg_client = MongoClient("mongodb://localhost:27017/")
mg_db = mg_client["PLP"]
mg_col = mg_db["COVID_FAQ"]

def prepare_inferrence_features(question, context):
   
    tokenized_example = tokenizer.encode_plus(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    #print("tokenized_example:{0}".format(tokenized_example))
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_example["example_id"] = []

    for i in range(len(tokenized_example["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_example["example_id"].append(sample_index)

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_example["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_example["offset_mapping"][i])
        ]

    return tokenized_example


def find_best_answer(question, context): 
  example_features=prepare_inferrence_features(question.strip(),context.strip())
  answer_model=saved_model(torch.tensor(example_features["input_ids"]), attention_mask=torch.tensor(example_features["attention_mask"])) 
  for i in range(len(example_features["offset_mapping"])):
    start_logits = answer_model.start_logits[i].detach().numpy()
    #print("start_logits:{0}".format(start_logits))
    end_logits = answer_model.end_logits[i].detach().numpy()
    #print("end_logits:{0}".format(end_logits))
    offset_mapping = example_features["offset_mapping"][i]
    #print("offset_mapping:{0}".format(offset_mapping))
    # The first feature comes from the first example. For the more general case, we will need to be match the example_id to
    # an example index 
    
    # Gather the indices the best start/end logits:
    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
    #print("start_indexes:{0}".format(start_indexes))
    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
    #print("end_indexes :{0}".format(end_indexes))
    valid_answers = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
            # to part of the input_ids that are not in the context.
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                #print("final start_index:{0}, final end index:{1}".format(start_index,end_index))
                
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char: end_char]
                    }
                )
    
  best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
  return best_answer["text"]


def get_context_by_question(question):
    #model = AutoModel.from_pretrained(model_checkpoint)
    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    print("question is:{0}".format(question))
    #print("question vector is:{0}".format(fe(question.strip())))
    print("squeeze:{0}".format(np.squeeze(fe(question.strip()))))
    valid_contexts=[]
    query_squeeze_array=np.squeeze(fe(question.strip()))
    query_question_vector =torch.tensor(query_squeeze_array,dtype=torch.float32)
    #query_question_vector = torch.tensor(fe(question.strip()))

    print("query vector:{0}".format(query_question_vector))
    query_question_sent=torch.mean(query_question_vector,dim=0)
    for doc in mg_col.find():
        searched_question_vector=torch.tensor(np.squeeze(doc["question_features"]))
        searched_question_sent=torch.mean(searched_question_vector,dim=0)
        sim_score=1-cosine(query_question_sent,searched_question_sent)
        valid_contexts.append(
            {
             "score": sim_score,
             "context":doc["answer"]
            }
            
        )
    #print(valid_contexts)
    best_context = sorted(valid_contexts, key=lambda x: x["score"], reverse=True)[0] 
    print('best_context:{0}'.format(best_context))
    return best_context["context"]

def generate_response(question):
    context=get_context_by_question(question)
    bestAnswer=find_best_answer(question,context)
    return bestAnswer   

#'''
if __name__ == "__main__":
 
    question="Who should receive a booster?"
    print("yy")
    tensor1 = torch.tensor(np.arange(5))
    print(tensor1)
    context=get_context_by_question(question)
    #print("The detected context is:{0}".format(context))
    bestAnswer=find_best_answer(question,context)
    print("The best generated answer is:")
    print(bestAnswer)
#'''


