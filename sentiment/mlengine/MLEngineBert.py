"""
Author: Lu ZhiPing
Email: lu.zhiping@u.nus.edu
"""
from transformers import DistilBertTokenizer
from onnxruntime import InferenceSession
import numpy as np


def softmax(x):
    return np.exp(x) / np.exp(x).sum()

class MlEngineBert:
    """
    Inference Engine using ONNX for optimized speed in CPU instance
    """
    def __init__(self):
        """
        Implement me
        """
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained("tokenizer")
            self.session = InferenceSession("torch-model.onnx")
        except FileNotFoundError:
            raise "Tokenizer and Model file not found"


    def __call__(self, *args, **kwargs):
        """
        Implement me
        """
        inputs = self.tokenizer(args[0].lower(), return_tensors="np")
        logit = self.session.run(None, input_feed=dict(inputs))
        prediction = softmax(logit[0]).round(4)
        return {
            "negative": prediction[0][0],
            "neutral": prediction[0][1],
            "positive": prediction[0][2]
        }


if __name__ == "__main__":
    text = 'HDB closes Bukit Merah branch office after second employee tests positive for Covid-19 https://t.co/hhbICSfy5o'
    engine = MlEngineBert()
    print(engine(text))
    # {'negative': 1e-04, 'neutral': 1e-04, 'positive': 0.9998}
