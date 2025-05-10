#!/bin/bash

# Set the relative path to the second repo (e.g., config files)
CONFIG_REPO="../../analysis-config-for-k-onda"

msg="$1"
if [ -z "$msg" ]; then
  echo "Usage: $0 \"Commit message\""
  exit 1
fi

# Stage and commit changes in this repo
git add -A
git commit -m "$msg"

# If the second repo exists, do the same there
if [ -d "$CONFIG_REPO/.git" ]; then
  (
    cd "$CONFIG_REPO" || exit
    git add -A
    git commit -m "$msg"
  )
else
  echo "⚠️  No Git repo found at $CONFIG_REPO — skipping secondary commit."
fi