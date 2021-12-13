from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from uuid import uuid4

import emoji
from transformers import AutoModelForMaskedLM, AutoTokenizer

app = FastAPI()

from pydantic import BaseModel
from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi_chameleon import template, global_init

global_init("./templates", auto_reload=True)  # False in prd


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

    emoji_txt = set(tokenizer.get_vocab().keys()).intersection(
        emoji.UNICODE_EMOJI_ENGLISH.values()
    )
    emoji_indices = tokenizer.encode(" ".join(emoji_txt), add_special_tokens=False)
    emoji_icons = [emoji.emojize(e) for e in emoji_txt]

    # calc raw proba for later lift
    input_ids = encode(tokenizer, "")
    outputs = model(input_ids)[0]
    logits = outputs[0, -2, :]  # last token is start of sentence token
    raw_proba = logits.softmax(dim=0)[emoji_indices]

    def predict(user_text):
        """predicts the answer on an given question and context. Uses encode and decode method from above"""
        # encode - predict - decode
        input_ids = encode(tokenizer, user_text)

        # predict
        outputs = model(input_ids)[0]
        logits = outputs[0, -2, :]  # last token is start of sentence token
        emoji_proba = logits.softmax(dim=0)[emoji_indices]
        lift = (emoji_proba / raw_proba).argsort(descending=True)
        return [emoji_icons[i] for i in lift]

    return predict


# initializes the pipeline
emoji_pipeline = serverless_pipeline()

# This depends stuff is needed so we can accept form encoded bodies
# https://stackoverflow.com/questions/61872923/supporting-both-form-and-json-encoded-bodys-with-fastapi#61881882
# htmx promises also to accept json-enc: https://htmx.org/extensions/json-enc/
# but it doesnt work https://github.com/bigskysoftware/htmx/issues/210
# @template(template_file="search-results.pt")
@app.post("/search", response_class=HTMLResponse)
# async def search(item: Item = Depends(Item.as_form)):
async def search(input_text: str = Form(...)):
    uuid = uuid4()
    answers = emoji_pipeline(input_text)
    # TODO: log(input_text, uuid)
    return "<ul>" + "".join([f"<li onclick=\"fnOnClick('{a}', '{uuid}', {i})\">{a}</li>" for i, a in enumerate(answers)]) + "</ul>"


async def click(uuid, emoji, index):
    pass
    # TODO: log(uuid, emoji, index)


@app.get("/")
@template(template_file="index.html")
async def index():
    return {"answers": None}
