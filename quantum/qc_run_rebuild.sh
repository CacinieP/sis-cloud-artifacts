#!/bin/sh
# Run the rebuild script with the qualtran interpreter chosen by qc_env.sh.
# argv: FACT_NPZ REF_NPZ OUT_DIR BROTA BATCH
QENV=$(cat /kaggle/working/qenv_path.txt)
if [ -z "$QENV" ] || [ "$QENV" = "NONE" ]; then
    echo "REBUILD_ENV_MISSING"
    exit 3
fi
exec "$QENV" /kaggle/working/qc_rebuild.py "$1" "$2" "$3" "$4" "$5"
