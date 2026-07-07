#!/usr/bin/env bash
# =============================================================================
# Verify + benchmark the Modal transcript-only fast path.
#
# Exercises what makes Modal-only transcription work:
#   1. warm ping actually PRELOADS Whisper (model_loaded == true)
#   2. keep-warm (scaledown_window) keeps the container alive so the follow-up
#      transcript call reuses it (real.task_id == warm.task_id, cold == false).
#      NB: Modal rejects client-set `modal-*` headers, so there is NO session
#      header — affinity here is best-effort keep-warm, not hard pinning.
#
# Prereqs: deploy the app first  ->  modal deploy main.py
#          jq + curl on PATH
#
# Usage:
#   ./bench_transcript.sh <bucket> <location> <filename> [modal_url]
#
# Example (a video already staged in R2/S3):
#   ./bench_transcript.sh logie-public "media/videos/toprated-videos" "24 inch knife holder.mp4"
# =============================================================================
set -uo pipefail

BUCKET="${1:?need <bucket>}"
LOCATION="${2:?need <location> (folder inside the bucket, no bucket name)}"
FILENAME="${3:?need <filename> e.g. video.mp4}"
URL="${4:-https://mentalizer--video-extractor-final-process.modal.run}"
MODEL="${MODAL_WHISPER_MODEL:-Systran/faster-distil-whisper-large-v3}"

echo "▶ model=$MODEL"
echo "▶ url=$URL"
echo

# post <json> -> sets globals BODY (response body) and CODE (http status).
# Prints the raw body + status and aborts if the body isn't valid JSON, so a
# platform error / timeout page is legible instead of a cryptic jq failure.
post() {
  local label="$1" json="$2" resp
  resp=$(curl -sS -w $'\n%{http_code}' --max-time 600 -X POST "$URL" \
    -H "content-type: application/json" -d "$json")
  CODE=$(printf '%s' "$resp" | tail -n1)
  BODY=$(printf '%s' "$resp" | sed '$d')
  if ! printf '%s' "$BODY" | jq -e . >/dev/null 2>&1; then
    echo "❌ $label did not return JSON (HTTP $CODE). Raw response:"
    echo "----------------------------------------------------------------------"
    printf '%s\n' "$BODY"
    echo "----------------------------------------------------------------------"
    exit 1
  fi
}

echo "── 1) warm ping (preload Whisper) ─────────────────────────────────────"
echo "   (first call after a redeploy may take a while — cold boot + model load/download)"
post "warm ping" "{\"warm\":true,\"preload_whisper\":true,\"whisper_model\":\"$MODEL\"}"
echo "$BODY" | jq '{status, task_id, cold, model_loaded, face_loaded, enter_seconds, ready_in_seconds}'
WARM_TASK=$(echo "$BODY" | jq -r '.task_id // "none"')
if [ "$(echo "$BODY" | jq -r '.model_loaded')" = "true" ]; then
  echo "✅ Whisper preloaded on the warm container"
elif [ "$(echo "$BODY" | jq -r '.status')" = "warm" ]; then
  echo "⚠️  warmed but model_loaded not true — deployed build may predate the instrumentation, or preload_whisper was dropped"
else
  echo "❌ unexpected warm response"
fi
echo

echo "── 2) transcript-only (should reuse the warm container) ───────────────"
START=$(python3 -c 'import time; print(time.time())')
post "transcript-only" "{\"bucket\":\"$BUCKET\",\"location\":\"$LOCATION\",\"filename\":\"$FILENAME\",\"transcript_only\":true,\"whisper\":\"always\",\"whisper_model\":\"$MODEL\",\"language\":\"en\"}"
END=$(python3 -c 'import time; print(time.time())')

echo "$BODY" | jq '{status, mode, task_id, cold, enter_seconds, processing_metrics}'
echo "── language routing ──"
echo "$BODY" | jq '.transcript_metadata.whisper // {}'
echo "── transcript (first 400 chars) ──"
echo "$BODY" | jq -r '.text // .error' | head -c 400; echo; echo

WALL=$(python3 -c "print(f'{$END - $START:.2f}')")
TR_TASK=$(echo "$BODY" | jq -r '.task_id // "none"')
MODE=$(echo "$BODY" | jq -r '.mode // "none"')
echo "⏱  end-to-end wall-clock: ${WALL}s"
[ "$MODE" = "transcript" ] && echo "✅ transcript-only fast path ran" || echo "⚠️  mode='$MODE' — deployed build may not honor transcript_only (ran the full pipeline)"
[ "$(echo "$BODY" | jq -r '.cold')" = "false" ] && echo "✅ warm hit (cold=false)" || echo "⚠️  cold start (cold=true) — container had scaled down; raise scaledown_window or set min_containers>=1"
if [ "$WARM_TASK" = "$TR_TASK" ] && [ "$WARM_TASK" != "none" ]; then
  echo "✅ reused the warm container ($TR_TASK) — keep-warm held across the two calls"
else
  echo "ℹ️  different container (warm=$WARM_TASK real=$TR_TASK) — expected only under concurrent load; set min_containers>=1 for a hard guarantee"
fi
echo
echo "Tip: cross-check the container lifecycle with:"
echo "  modal app logs video-extractor-final -f --show-container-id --search WHISPER_PRELOADED --since 15m"
