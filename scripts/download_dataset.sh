#!/bin/bash
# Downloads DIV2K validation set (bicubic X4, 100 PNG images)
set -e

URL="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
ZIPFILE="data/DIV2K_valid_LR_bicubic_X4.zip"
EXTRACT_DIR="data/DIV2K_valid_LR_bicubic/X4"

if [ -d "$EXTRACT_DIR" ] && [ "$(ls -1 "$EXTRACT_DIR"/*.png 2>/dev/null | wc -l)" -ge 90 ]; then
    echo "[INFO] Dataset already extracted at $EXTRACT_DIR"
    exit 0
fi

mkdir -p data

if [ ! -f "$ZIPFILE" ]; then
    echo "[INFO] Downloading DIV2K valid LR bicubic X4 ..."
    wget -q --show-progress -O "$ZIPFILE" "$URL" || curl -L -o "$ZIPFILE" "$URL"
fi

echo "[INFO] Extracting ..."
unzip -o "$ZIPFILE" -d data/
echo "[INFO] Done. Images in $EXTRACT_DIR"
