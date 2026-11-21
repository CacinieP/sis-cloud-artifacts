#!/bin/sh
# Install qualtran into /opt/conda (when present) or a uv python 3.12 venv.
# On success prints "QENV=<interpreter path>" and writes that path to
# /kaggle/working/qenv_path.txt for the rebuild wrapper.
set -x

if [ -x /opt/conda/bin/python ]; then
    /opt/conda/bin/python -m pip install -q qualtran cirq
    if /opt/conda/bin/python -c "import qualtran" 2>/dev/null; then
        echo "QENV=/opt/conda/bin/python"
        echo "/opt/conda/bin/python" > /kaggle/working/qenv_path.txt
        exit 0
    fi
fi

/usr/bin/python3 -m pip install -q uv

for U in /usr/local/bin/uv /usr/bin/uv /root/.local/bin/uv; do
    if [ -x "$U" ] || [ -f "$U" ]; then
        "$U" venv --python 3.12 /kaggle/working/qenv
        if [ -x /kaggle/working/qenv/bin/python3.12 ] || [ -x /kaggle/working/qenv/bin/python ]; then
            "$U" pip install --python /kaggle/working/qenv/bin/python qualtran cirq numpy
            if /kaggle/working/qenv/bin/python -c "import qualtran" 2>/dev/null; then
                echo "QENV=/kaggle/working/qenv/bin/python"
                echo "/kaggle/working/qenv/bin/python" > /kaggle/working/qenv_path.txt
                exit 0
            fi
        fi
    fi
done

echo "QENV=NONE"
exit 1
