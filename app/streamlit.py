import emoji
import streamlit as st
import my_module


@st.cache
def get_base():
    return my_module.get_emoji("", model, tokenizer)


load_model = st.cache(allow_output_mutation=True)(my_module.load_model)

model, tokenizer = load_model()
base_output = get_base()

st.write("# Emoji Recommender")

user_text = st.text_input("Write here:", "trick or treat")

# removing some non-emotional characters seems to help
cleaned_text = emoji.demojize(user_text.strip(",.;:"))
output = my_module.get_emoji(cleaned_text, model, tokenizer)
lift = (output.sort_index() / base_output.sort_index()).sort_values(ascending=False)
st.write("## " + emoji.emojize("".join(lift.index)))


st.markdown(
    "Made by [Rens](https://twitter.com/R_Dimm) with help from [VinAIResearch](https://github.com/VinAIResearch/BERTweet)"
)

