.PHONY: build run train train-bg deploy down clean-results clean-weights logs stop-train

# Get IDs from the host environment
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

# Define the base docker commands as macros
DC := USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose
DC_RUN := $(DC) run --rm flow_matching

# Default config file (override with: make train CONFIG=path/to/config.yaml)
CONFIG ?= configs/lasa.yaml

build:
	$(DC) build

run:
	$(DC_RUN) bash

train:
	$(DC_RUN) python scripts/train.py --config $(CONFIG)

train-bg:
	$(DC) run -d --rm --name flow_training flow_matching python scripts/train.py --config $(CONFIG)
	@echo "Training started in the background. Use 'make logs' to view output."

logs:
	docker logs -f flow_training

stop-train:
	docker stop flow_training

deploy:
	$(DC_RUN) python scripts/deploy_lasa.py --config $(CONFIG)

down:
	$(DC) down

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