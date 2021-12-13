#!/usr/bin/env bash
docker image build --tag emoji .
# gcloud builds submit --tag gcr.io/rens-sandbox/emoji