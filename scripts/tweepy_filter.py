from os import environ
import tweepy
import time

import json

g = []


class TweetPrinter(tweepy.Stream):
    def on_data(self, data):
        # Streaming API. Streaming API listens for live tweets
        data = json.loads(data)
        if "text" in data:
            print(data["text"])
            print("==================")
            g.append(data)
        return True

    def on_status(self, status):
        print(status.id)


printer = TweetPrinter(
    environ["CONSUMER_KEY"],
    environ["CONSUMER_SECRET"],
    environ["ACCESS_TOKEN"],
    environ["ACCESS_SECRET"],
)

printer.filter(track=["🚀"], languages=["en"])
