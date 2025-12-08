#!/bin/bash

echo "=== Clearing old logs ==="
adb logcat -c

echo "=== Starting crash monitor ==="
echo "Now launch your app and watch for errors below..."
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="

# Monitor for crashes with full stack traces
adb logcat | grep --line-buffered -A 50 "AndroidRuntime\|FATAL\|Exception\|Error" | grep --line-buffered -v "chatty\|native\|DEBUG"

