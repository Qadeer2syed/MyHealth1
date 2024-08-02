# Makefile for Python Poetry Project

# Install project dependencies
install:
	pipx install poetry
	poetry shell
	poetry install

# Run the project
run_demo:
	python myhealth/myHealth.py path/to/image.jpeg

# Test the project
test:
	poetry run pytest

# Build the project
build:
	poetry build

# Clean up the project
clean:
	rm -rf .build
