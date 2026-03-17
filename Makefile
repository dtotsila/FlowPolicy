.PHONY: build run down

build:
	docker compose build

run:
	docker compose run --rm flow_matching bash

down:
	docker compose down