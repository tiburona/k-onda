.PHONY: check ready fix test lint lint-fix notebook-clean-check notebook-clean pre-commit binder-requirements

check: lint notebook-clean-check test

ready: fix binder-requirements pre-commit check

fix: lint-fix notebook-clean

test:
	.venv/bin/python -m pytest tests

lint:
	uv run --locked ruff check .

lint-fix:
	uv run --locked ruff check . --fix

notebook-clean-check:
	nbstripout --verify demo/k_onda_demo.ipynb

notebook-clean:
	nbstripout demo/k_onda_demo.ipynb

pre-commit:
	pre-commit run --all-files

binder-requirements:
	uv export \
		--format requirements.txt \
		--extra demo \
		--no-dev \
		--no-hashes \
		--no-header \
		--frozen \
		--output-file requirements.txt
