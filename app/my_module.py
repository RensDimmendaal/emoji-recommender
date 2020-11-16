from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import numpy as np


def load_model():
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    return model, tokenizer


def filter_characters(df):
    return (
        df.loc[lambda d: d["token_str"].str.startswith(":")]
        .loc[lambda d: d["token_str"].str.endswith(":")]
        .loc[lambda d: ~d["token_str"].str.contains("regional_indicator_symbol_letter")]
    )


def get_emoji(text, model, tokenizer):
    input_ids = tokenizer.encode(text + " <mask>", return_tensors="pt")

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
            {"score": v, "token": p, "token_str": tokenizer.convert_ids_to_tokens(p),}
        )
    return (
        pd.DataFrame(result)
        .pipe(filter_characters)
        .assign(score=lambda d: d["score"] / d["score"].sum())
        .set_index("token_str")["score"]
    )
