binder-requirements:
	uv export \
		--format requirements.txt \
		--extra demo \
		--no-dev \
		--no-hashes \
		--frozen \
		--output-file requirements.txt
