#!/bin/bash
# Periodically log Optuna trial progress and best metrics.
#
# Usage: ./monitor_optuna_progress.sh [log_file] [interval_seconds]

DB_PATH="_optuna/noaug.db"
LOG_FILE="${1:-hpo_optuna_progress.log}"
INTERVAL="${2:-60}"

if [[ ! -f "$DB_PATH" ]]; then
  echo "Optuna database not found at ${DB_PATH}" >&2
  exit 1
fi

export LC_ALL=C
echo "=== Optuna Progress Monitor started at $(date) ===" >> "${LOG_FILE}"
echo "Database: ${DB_PATH}" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

while true; do
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  # Total trials
  total="$(sqlite3 "${DB_PATH}" "SELECT COUNT(*) FROM trials;")"

  # State breakdown
  state_rows="$(sqlite3 "${DB_PATH}" "SELECT state, COUNT(*) FROM trials GROUP BY state;")"
  state_summary=""
  while IFS='|' read -r state count; do
    [[ -z "$state" ]] && continue
    state_summary+="${state}=${count} "
  done <<< "${state_rows}"
  state_summary="${state_summary%" "}"

  # Best completed trial
  best_row="$(sqlite3 "${DB_PATH}" "SELECT t.trial_id, tv.value FROM trials t JOIN trial_values tv ON t.trial_id = tv.trial_id WHERE t.state = 'COMPLETE' ORDER BY tv.value DESC LIMIT 1;")"
  best_trial="n/a"
  best_value="n/a"
  if [[ -n "$best_row" ]]; then
    best_trial="${best_row%%|*}"
    best_value="${best_row##*|}"
  fi

  echo "[${timestamp}] total=${total} states=${state_summary} best_trial=${best_trial} best_value=${best_value}" >> "${LOG_FILE}"

  sleep "${INTERVAL}"
done
