.PHONY: build run down clean-results clean-weights

# Get IDs from the host environment
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

build:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose build

run:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose run --rm flow_matching bash

down:
	docker compose down

clean-results:
	rm -rf results/*

clean-weights:
	@read -p "Are you sure you want to delete all weights? This action cannot be undone. (y/n) " answer; \
	if [ "$$answer" = "y" ]; then \
		rm -rf weights/* ; \
		echo "All weights have been deleted."; \
	else \
		echo "Operation cancelled. No weights were deleted."; \
	fi