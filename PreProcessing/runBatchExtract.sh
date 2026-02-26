#!/bin/bash
# Print script start
echo "Starting batch extraction..."

# ------------------------------------------------------------------------------
# 1) Initialize conda in non-interactive shell via shell hook
# ------------------------------------------------------------------------------

if command -v conda >/dev/null 2>&1; then
    # This sets up the 'conda' function in a non-interactive bash
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: 'conda' command not found in PATH" >&2
    exit 1
fi


conda activate hugoEnv || { echo "ERROR: failed to activate hugoEnv"; exit 1; }

# ------------------------------------------------------------------------------
# 2) Change to the directory containing your script
# ------------------------------------------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"


python batchExtract2.py \
    --pdf-dir "../SourceArticles/PDFs" \
    --output-dir "../SourceArticles/Extractions" \
    --page-limit 750 \
    --lang en


python batchParse3.py
python batchMetaData.py
python batchScan2.py
python batchScan2.5_figureInfo.py


conda deactivate
