#!/bin/bash

# List of relative paths to repositories or directories to commit
REPOS=(
  "."  # Current repo
  "../k-onda-analysis"
)

msg="$1"
if [ -z "$msg" ]; then
  echo "Usage: $0 \"Commit message\""
  exit 1
fi

for REPO in "${REPOS[@]}"; do
  if [ -d "$REPO/.git" ]; then
    echo "🔄 Committing changes in $REPO..."
    (
      cd "$REPO" || exit
      git add -A
      git commit -m "$msg"
    )
  else
    echo "⚠️  No Git repo found at $REPO — skipping."
  fi
done