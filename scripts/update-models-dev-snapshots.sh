#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

API=$(curl -fsSL https://models.dev/api.json)

printf '%s' "$API" | jq '.openai'    > crates/anyllm-openai/models-dev.json
printf '%s' "$API" | jq '.anthropic' > crates/anyllm-anthropic/models-dev.json
printf '%s' "$API" | jq '.google'    > crates/anyllm-gemini/models-dev.json
