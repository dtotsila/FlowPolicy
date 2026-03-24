.PHONY: build run down

# Get IDs from the host environment
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

build:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose build

run:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose run --rm flow_matching bash

down:
	docker compose down