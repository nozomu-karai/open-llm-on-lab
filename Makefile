lint:
	flake8 .
	isort --check --diff .
	black --check .

format:
	isort .
	black .
