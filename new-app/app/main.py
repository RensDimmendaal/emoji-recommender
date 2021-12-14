from json import dumps
from logging import getLogger, INFO
from uuid import uuid4
import numpy as np

import emoji
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi_chameleon import template, global_init
from transformers import AutoModelForMaskedLM, AutoTokenizer

app = FastAPI()

global_init("./app/templates/", auto_reload=True)  # False in prd
app.mount("/app/static", StaticFiles(directory="./app/static"), name="static")

logger = getLogger("gunicorn.error")
logger.setLevel(INFO)


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
        lift = (emoji_proba / raw_proba)
        sort = lift.argsort(descending=True)
        return np.array(emoji_icons)[sort], lift.detach().numpy()[sort]

    return predict


def opacity(x, x_break=1, opacity_break=0.8):
    opacities = np.zeros(len(x))
    opacities[x > x_break] = opacity_break + (1-opacity_break) * (x[x > x_break] - x_break) / x.max()
    opacities[x <= x_break] = (x[x <= x_break] / x_break) * opacity_break
    return opacities


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
    uuid = str(uuid4())
    emojis, scores = emoji_pipeline(input_text)
    opacities = np.round(opacity(scores, x_break=1, opacity_break=0.8), 2)

    logger.info(f"SEARCH|{dumps(dict(input_text=input_text, uuid=uuid))}")
    return (
        "<div class='emoji-results'>"
        + "".join(
            [
                f"<a class='emoji' style='opacity:{opacity}' onclick=\"fnOnClick('{emoji}', '{uuid}', {i})\">{emoji}</a>"
                for i, (emoji, opacity) in enumerate(zip(emojis, opacities))
            ]
        )
        + "</div>"
    )


@app.post("/click", response_class=HTMLResponse)
async def click(uuid: str = Form(...), emoji: str = Form(...), index: str = Form(...)):
    logger.info(f"CLICK|{dumps(dict(uuid=uuid, index=index, emoji=emoji))}")


@app.get("/")
@template(template_file="index.html")
async def index():
    return {"answers": None}
