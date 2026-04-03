#!/bin/bash
# Convert overheard.svg → overheard.icns for macOS app bundle
# Requires: rsvg-convert (brew install librsvg) and iconutil

set -e
cd "$(dirname "$0")"

echo "Rendering SVG at multiple sizes..."
ICONSET="overheard.iconset"
mkdir -p "$ICONSET"

for size in 16 32 64 128 256 512 1024; do
    rsvg-convert -w $size -h $size overheard.svg -o "$ICONSET/icon_${size}x${size}.png"
done

# macOS iconutil needs @2x variants
cp "$ICONSET/icon_32x32.png"   "$ICONSET/icon_16x16@2x.png"
cp "$ICONSET/icon_64x64.png"   "$ICONSET/icon_32x32@2x.png"
cp "$ICONSET/icon_256x256.png" "$ICONSET/icon_128x128@2x.png"
cp "$ICONSET/icon_512x512.png" "$ICONSET/icon_256x256@2x.png"
cp "$ICONSET/icon_1024x1024.png" "$ICONSET/icon_512x512@2x.png"

echo "Building .icns..."
iconutil -c icns "$ICONSET" -o overheard.icns

echo "Done → icon/overheard.icns"
