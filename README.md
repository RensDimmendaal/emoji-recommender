# Emoji Recommender

As someone who writes to friends, colleagues, twitter, and blogs
I want to add joy by including emoji
I need to get suggestions on what emoji to include
What's holding me back is that there are too many emoji to know all by heart
And the search in apple or on websites such as find emoji is too literal.

I would like to get more meaningful suggestions

A website where I can paste a sentence and get suggestions back would be a great start


![](example.png)

## 👉 Approach

Use https://github.com/VinAIResearch/BERTweet with the MaskLM task to find emoji that fit a sentence

Initial implementation: https://colab.research.google.com/drive/1UB55pRJcK329ASqt1VaWxHbdNHocnlu3#scrollTo=szd-681JyfeW

## 👀 Watch out for

* Some emoji are not included in that vocab: e.g. spider and spiderweb
* Inference might be slow and/or expensive
