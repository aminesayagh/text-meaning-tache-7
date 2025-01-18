#!/bin/bash

# Create docs directory if it doesn't exist
mkdir -p docs

# Generate tree structure with specific ignores
TREE_OUTPUT=$(tree -a -I 'node_modules|.git|.next|dist|.turbo|.cache|.vercel|coverage' \
     --dirsfirst \
     --charset=ascii)

{
  echo "# Project Tree Structure"
  echo "\`\`\`plaintext"
  echo "$TREE_OUTPUT"
  echo "\`\`\`"
} > docs/doc-project-tree.md

cw doc \
    --pattern ".yml|.conf|.json|.sh|.py" \
    --exclude ".pyc|.txt|.ipynb|.ipynb" \
    --output "docs/doc.md" \
    --compress false


