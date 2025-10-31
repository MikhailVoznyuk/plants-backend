@echo off
setlocal ENABLEDELAYEDEXPANSION
py -m pip install -U requests pillow >NUL 2>&1
docker build -t treehealth-gpu -f Dockerfile.gpu . || goto :err
docker compose -f docker-compose.gpu.yml up -d --build || goto :err
py tools\e2e_test.py --url http://localhost:8000 --report smoke_report_gpu.json || goto :err
echo [OK] GPU smoke passed. Report: smoke_report_gpu.json
exit /b 0
:err
echo [FAIL] GPU smoke failed.
exit /b 1
