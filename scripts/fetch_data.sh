#!/usr/bin/env bash
# Download a bounded slice of OpenSanctions into data/raw/.
# By default fetch the pinned versions that eval_baseline.json was measured on, so the CI
# quality gate measures code changes, not upstream data changes. OPENSANCTIONS_LATEST=1
# resolves the newest versions from the index instead (then re-baseline).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

INDEX=$(mktemp)
if [ "${OPENSANCTIONS_LATEST:-0}" = 1 ]; then
  curl -fsSL "https://data.opensanctions.org/datasets/latest/index.json" -o "$INDEX"
fi

pinned () {  # pinned <dataset> -> version the baseline was measured on
  case "$1" in
    us_ofac_cons)   echo 20260920223501-krw ;;
    gb_hmt_invbans) echo 20260920185601-kko ;;
    us_ofac_sdn)    echo 20260920221915-jzv ;;
  esac
}

fetch () {  # fetch <dataset> <max_lines>
  local name="$1" limit="${2:-0}"
  local url
  if [ "${OPENSANCTIONS_LATEST:-0}" != 1 ]; then
    url="https://data.opensanctions.org/artifacts/$name/$(pinned "$name")/entities.ftm.json"
  else
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
  fi
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
