from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_model(model):
    """Loads model from Hugginface model hub"""
    try:
        model = AutoModelForMaskedLM.from_pretrained(model)
        model.save_pretrained("./model")
    except Exception as e:
        raise (e)


def get_tokenizer(tokenizer):
    """Loads tokenizer from Hugginface model hub"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.save_pretrained("./model")
    except Exception as e:
        raise (e)


if __name__ == "__main__":
    model = "vinai/bertweet-base"
    get_model(model)
    get_tokenizer(model)
