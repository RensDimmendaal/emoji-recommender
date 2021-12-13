#!/usr/bin/env bash
docker image build --tag emoji .
# gcloud config set project rens-sandbox
# gcloud builds submit --tag gcr.io/rens-sandbox/emoji