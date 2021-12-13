.PHONY: build
build:
	docker build -t "emoji" .

.PHONY: run
run: build
	docker run -p 8080:8080 --rm -it "emoji"

.PHONY: deploy
deploy:
	gcloud --project rens-sandbox \
    run deploy emoji \
		--source . \
		--platform managed \
		--region europe-west4 \
		--allow-unauthenticated \
		--max-instances=1 \
    --cpu=2 \
		--memory=8Gi