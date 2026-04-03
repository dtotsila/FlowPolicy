.PHONY: help build run train train-bg deploy down clean-results clean-weights logs stop-train data-prepare viz-data

# Get IDs from the host environment
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

# Define the base docker commands as macros
DC := USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose
DC_RUN := $(DC) run --rm flow_matching

# Default config file (override with: make train CONFIG=path/to/config.yaml)
CONFIG ?= configs/lasa.yaml

.DEFAULT_GOAL := help

help: ## Show this help message and list all available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

data-prepare: ## Build the HDF5 dataset from the raw LASA data
	$(DC_RUN) python scripts/create_hdf5.py --output data/lasa_vision.hdf5

viz-data: ## Generate a grid overview of the HDF5 dataset in the results folder
	$(DC_RUN) python scripts/visualize_hdf5.py --dataset data/lasa_vision.hdf5

build: ## Build the Docker image
	$(DC) build

run: ## Open an interactive bash shell inside the container
	$(DC_RUN) bash

train: ## Run the training script interactively (blocks terminal)
	$(DC_RUN) python scripts/train.py --config $(CONFIG)

train-bg: ## Run the training script in the background (detached mode)
	$(DC) run -d --rm --name flow_training flow_matching python scripts/train.py --config $(CONFIG)
	@echo "Training started in the background. Use 'make logs' to view output."

logs: ## Follow the logs of the background training container
	docker logs -f flow_training

stop-train: ## Stop the background training container gracefully
	docker stop flow_training

deploy: ## Run the deployment/rollout script to test the model
	$(DC_RUN) python scripts/deploy_lasa.py --config $(CONFIG)

down: ## Tear down the Docker Compose environment
	$(DC) down

clean-results: ## Delete all generated plots and rollouts in the results/ directory
	rm -rf results/*

clean-weights: ## (Interactive) Delete all saved model checkpoints in the weights/ directory
	@read -p "Are you sure you want to delete all weights? This action cannot be undone. (y/n) " answer; \
	if [ "$$answer" = "y" ]; then \
		rm -rf weights/* ; \
		echo "All weights have been deleted."; \
	else \
		echo "Operation cancelled. No weights were deleted."; \
	fi