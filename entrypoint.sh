#!/bin/sh
set -e

echo "=== Athlink CV Service Starting ==="
echo "PORT=${PORT:-not set}"

python startup_check.py

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8005}"
