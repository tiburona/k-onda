binder-requirements:
	uv export \
		--format requirements.txt \
		--extra demo \
		--no-dev \
		--no-hashes \
		--no-header \
		--frozen \
		--output-file requirements.txt
