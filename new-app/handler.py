import json

import emoji
import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer


def encode(tokenizer, user_text):
    """encodes the question and context with a given tokenizer"""
    text = emoji.demojize(user_text.strip(",.;:"))
    encoded = tokenizer.encode(text + " <mask>", return_tensors="pt")

    return encoded


def decode(tokenizer, token):
    """decodes the tokens to the answer with a given tokenizer"""
    answer_tokens = tokenizer.convert_ids_to_tokens(token, skip_special_tokens=True)
    return tokenizer.convert_tokens_to_string(answer_tokens)


def serverless_pipeline(model_path="./model"):
    """Initializes the model and tokenzier and returns a predict function that ca be used as pipeline"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    def predict(user_text):
        """predicts the answer on an given question and context. Uses encode and decode method from above"""
        # encode - predict - decode
        input_ids = encode(tokenizer, user_text)

        # predict
        outputs = model(input_ids)[0]
        logits = outputs[0, -2, :]  # last token is start of sentence token
        probs = logits.softmax(dim=0)

        values, predictions = probs.topk(tokenizer.vocab_size)

        result = []
        for v, p in zip(values.tolist(), predictions.tolist()):
            tokens = input_ids.numpy()
            tokens[-1] = p
            # TODO: filter out all non-emoji tokens...
            # Filter padding out:
            tokens = tokens[np.where(tokens != tokenizer.pad_token_id)]
            result.append(
                {
                    "score": v,
                    "token": p,
                    "token_str": tokenizer.convert_ids_to_tokens(p),
                }
            )
        return result

    return predict


# initializes the pipeline
emoji_pipeline = serverless_pipeline()


def handler(event, context):
    try:
        print(event)
        print(context)
        # loads the incoming event into a dictonary
        body = json.loads(event["body"])
        # uses the pipeline to predict the answer
        answer = emoji_pipeline(user_text=body["user_text"])
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"top_emoji": answer}),
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(e)}),
        }
