#!/usr/bin/env bash
# Unload any lemonade/llama model that respawns and eats GPU memory, repeatedly.
# Run alongside JEPA training. Stops after $DURATION seconds or when killed.
DURATION="${1:-43200}"     # default 12h
INTERVAL="${2:-120}"       # check every 2 min
END=$(( $(date +%s) + DURATION ))
echo "[lemon-keeper] watching for lemonade respawns for ${DURATION}s..."
while [ "$(date +%s)" -lt "$END" ]; do
  # find the model that is loaded and unload it
  for m in "gemma-4-12b-it" "Gemma-4-12B-it-GGUF:latest" "Devstral-Small-2-24B-Instruct-2512-GGUF:Q4_K_M"; do
    curl -s --max-time 3 http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null 2>&1
  done
  sleep "$INTERVAL"
done
echo "[lemon-keeper] done."
