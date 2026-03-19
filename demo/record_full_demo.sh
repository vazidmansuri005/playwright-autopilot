#!/bin/bash
# Records full screen (terminal + browser) while running the autopilot demo.
#
# SAFE: API key is read from ~/.autopilot_key file, NEVER shown on screen.
#
# Setup (once):
#   echo "your-api-key" > ~/.autopilot_key
#
# Usage:
#   bash demo/record_full_demo.sh

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTPUT_DIR="demo/output"
mkdir -p "$OUTPUT_DIR"
VIDEO_FILE="$OUTPUT_DIR/full_demo.mp4"

# Read API key from file (never shown on screen)
if [ -f ~/.autopilot_key ]; then
  export ANTHROPIC_API_KEY=$(cat ~/.autopilot_key)
elif [ -n "$ANTHROPIC_API_KEY" ]; then
  : # Already set in environment
else
  echo "  ERROR: No API key found."
  echo "  Run: echo 'your-key' > ~/.autopilot_key"
  exit 1
fi

echo ""
echo "  ============================================"
echo "  playwright-autopilot — LinkedIn Demo"
echo "  ============================================"
echo ""
echo "  Recording screen to: $VIDEO_FILE"
echo "  API key: loaded from ~/.autopilot_key"
echo ""
echo "  Arrange: Terminal LEFT, Browser will open RIGHT"
echo "  Starting in 5 seconds..."
sleep 5

# Start screen recording (no key visible anywhere)
ffmpeg -f avfoundation \
  -framerate 30 \
  -capture_cursor 1 \
  -i "2:none" \
  -vf "scale=1920:1080" \
  -c:v libx264 \
  -preset fast \
  -crf 23 \
  -pix_fmt yuv420p \
  -y "$VIDEO_FILE" &

FFMPEG_PID=$!
sleep 2

echo "  Recording started."
echo ""

# Run the demo (key is in env, not on command line)
python demo/record_demo.py

sleep 3

# Stop recording
kill -INT $FFMPEG_PID 2>/dev/null || true
wait $FFMPEG_PID 2>/dev/null || true

if [ -f "$VIDEO_FILE" ]; then
  SIZE=$(du -h "$VIDEO_FILE" | cut -f1)
  echo ""
  echo "  Video: $VIDEO_FILE ($SIZE)"
  echo "  Upload to LinkedIn directly (MP4 supported)."
fi
