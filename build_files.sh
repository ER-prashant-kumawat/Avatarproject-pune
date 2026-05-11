#!/usr/bin/env bash
# Vercel build script — collect static files for WhiteNoise
set -e
echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Collecting static files ==="
python manage.py collectstatic --no-input

echo "=== Build complete ==="
