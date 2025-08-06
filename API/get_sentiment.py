from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession("sequence_classifier.onnx")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 128

class ReviewRequest(BaseModel):
    review: str

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@app.get("/")
def read_root():
    return {"message": "Sentiment analysis model is up and running! Have a great Day XD"}


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    try:
        review_text = request.review.strip()
        if not review_text:
            return {"error": "Review text cannot be empty."}

        output = tokenizer(review_text, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors="pt")
        inputs = {
            "input_ids": output["input_ids"].numpy().astype(np.int64),
            "attention_mask": output["attention_mask"].numpy().astype(np.float32)
        }

        logits = session.run(None, inputs)[0]
        probs = softmax(logits[0])

        return {
            "Negative": float(probs[0]),
            "Positive": float(probs[1])
        }

    except Exception as e:
        return {"error": str(e)}
