#!/usr/bin/env bash
# Download a bounded slice of OpenSanctions into data/raw/.
# The index lists every dataset with a versioned artefact URL, so resolve it at runtime
# rather than hardcoding a build that will 404 next week.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

INDEX=$(mktemp)
curl -fsSL "https://data.opensanctions.org/datasets/latest/index.json" -o "$INDEX"

fetch () {  # fetch <dataset> <max_lines>
  local name="$1" limit="${2:-0}"
  local url
  url=$(python3 -c "
import json,sys
idx=json.load(open('$INDEX'))
for d in idx.get('datasets',[]):
    if d.get('name')=='$name':
        for r in d.get('resources',[]):
            if r.get('name')=='entities.ftm.json':
                print(r.get('url') or ''); break
        break
")
  [ -n "$url" ] || { echo "no artefact for $name"; return 1; }
  echo "  $name <- $url"
  local out="data/raw/${name}.jsonl"
  if [ "$limit" -gt 0 ]; then
    # head closes the pipe once it has its lines, so curl dies with SIGPIPE and exit 23.
    # That is the expected way to take a slice, not a failure: disable pipefail around it
    # and check the result instead.
    set +o pipefail
    curl -fsSL "$url" | head -n "$limit" > "$out"
    set -o pipefail
  else
    curl -fsSL "$url" -o "$out"
  fi
  [ -s "$out" ] || { echo "empty download for $name"; return 1; }
}

fetch us_ofac_cons 0
fetch gb_hmt_invbans 0
fetch us_ofac_sdn 25000     # slice: the full SDN list is ~70k entities

wc -l data/raw/*.jsonl
