#!/bin/bash
# Downloads CImg.h single-header library into include/

set -e

CIMG_URL="https://raw.githubusercontent.com/GreycLab/CImg/master/CImg.h"
DEST="include/CImg.h"

if [ -f "$DEST" ]; then
    echo "[INFO] CImg.h already exists at $DEST"
    exit 0
fi

echo "[INFO] Downloading CImg.h v${CIMG_VERSION} ..."
mkdir -p include
wget -q --show-progress -O "$DEST" "$CIMG_URL" || curl -L -o "$DEST" "$CIMG_URL"
echo "[INFO] Done: $DEST"
