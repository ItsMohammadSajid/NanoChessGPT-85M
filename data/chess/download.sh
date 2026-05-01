#!/bin/bash
# ============================================================
# Lichess Elite Database — Full Download Script
# Source: https://database.nikonoel.fr/
# Filter: 2400+ vs 2200+ (2020-2021), 2500+ vs 2300+ (2022+)
# Format: .zip files containing PGN
# ============================================================

set -e  # Exit on any error

DOWNLOAD_DIR="./raw_zips"
mkdir -p "$DOWNLOAD_DIR"

echo "============================================"
echo " Lichess Elite Database Downloader"
echo " Total: ~66 months (Jun 2020 - Nov 2025)"
echo " Estimated size: ~6-7 GB (compressed)"
echo "============================================"

BASE_URL="https://database.nikonoel.fr"

# List of all months to download
MONTHS=(
    # 2020 (2400+ vs 2200+)
    "2020-06" "2020-07" "2020-08" "2020-09" "2020-10" "2020-11" "2020-12"
    # 2021 (2400+ vs 2200+, Dec onwards: 2500+ vs 2300+)
    "2021-01" "2021-02" "2021-03" "2021-04" "2021-05" "2021-06"
    "2021-07" "2021-08" "2021-09" "2021-10" "2021-11" "2021-12"
    # 2022 (2500+ vs 2300+)
    "2022-01" "2022-02" "2022-03" "2022-04" "2022-05" "2022-06"
    "2022-07" "2022-08" "2022-09" "2022-10" "2022-11" "2022-12"
    # 2023
    "2023-01" "2023-02" "2023-03" "2023-04" "2023-05" "2023-06"
    "2023-07" "2023-08" "2023-09" "2023-10" "2023-11" "2023-12"
    # 2024
    "2024-01" "2024-02" "2024-03" "2024-04" "2024-05" "2024-06"
    "2024-07" "2024-08" "2024-09" "2024-10" "2024-11" "2024-12"
    # 2025 (up to November, latest available)
    "2025-01" "2025-02" "2025-03" "2025-04" "2025-05" "2025-06"
    "2025-07" "2025-08" "2025-09" "2025-10" "2025-11"
)

TOTAL=${#MONTHS[@]}
COUNT=0
FAILED=()

for MONTH in "${MONTHS[@]}"; do
    COUNT=$((COUNT + 1))
    FILENAME="lichess_elite_${MONTH}.zip"
    FILEPATH="${DOWNLOAD_DIR}/${FILENAME}"
    URL="${BASE_URL}/${FILENAME}"

    echo ""
    echo "[${COUNT}/${TOTAL}] Downloading: ${FILENAME}"

    if [ -f "$FILEPATH" ]; then
        echo "  ✓ Already exists, skipping."
        continue
    fi

    # Download with retry (3 attempts)
    if wget --tries=3 --timeout=60 --progress=bar:force \
            -O "$FILEPATH" "$URL" 2>&1; then
        SIZE=$(du -sh "$FILEPATH" | cut -f1)
        echo "  ✓ Downloaded: ${SIZE}"
    else
        echo "  ✗ FAILED: ${URL}"
        rm -f "$FILEPATH"  # Remove incomplete file
        FAILED+=("$MONTH")
    fi
done

echo ""
echo "============================================"
echo " Download Complete!"
echo " Success: $((TOTAL - ${#FAILED[@]})) / ${TOTAL}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Failed months: ${FAILED[*]}"
    echo " Re-run the script to retry failed downloads."
fi

echo ""
echo " Next step: Run prepare.py to process the data"
echo "   python prepare.py --input_dir=./raw_zips"
echo "============================================"
