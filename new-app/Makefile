.PHONY: build
build:
	docker build -t "emoji" .

.PHONY: run
run: build
	docker run -p 8080:8080 --rm -it "emoji"

.PHONY: cloudbuild
cloudbuild:
	gcloud --project rens-sandbox builds submit --tag gcr.io/rens-sandbox/emoji

.PHONY: deploy
deploy:
	gcloud --project rens-sandbox \
    run deploy emoji \
		--image gcr.io/rens-sandbox/emoji \
		--platform managed \
		--port 8080 \
		--region europe-west4 \
		--allow-unauthenticated \
		--max-instances=1 \
    	--cpu=2 \
		--memory=8Gi