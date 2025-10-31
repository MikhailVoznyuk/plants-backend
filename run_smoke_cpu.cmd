@echo off
setlocal ENABLEDELAYEDEXPANSION
py -m pip install -U requests pillow >NUL 2>&1
docker build -t treehealth-cpu -f Dockerfile.cpu . || goto :err
docker compose -f docker-compose.cpu.yml up -d --build || goto :err
py tools\e2e_test.py --url http://localhost:8000 --report smoke_report_cpu.json || goto :err
echo [OK] CPU smoke passed. Report: smoke_report_cpu.json
exit /b 0
:err
echo [FAIL] CPU smoke failed.
exit /b 1
