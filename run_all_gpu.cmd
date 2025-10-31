@echo off
setlocal
echo === Building GPU image ===
docker build -t treehealth-gpu -f Dockerfile.gpu . || goto :err
echo === Starting GPU compose ===
docker compose -f docker-compose.gpu.yml up -d --build || goto :err
echo === Wait 5s for warmup ===
ping -n 6 127.0.0.1 >NUL
echo === Running suite ===
call tools\run_suite.cmd || goto :err
echo [OK] GPU pipeline passed.
exit /b 0
:err
echo [FAIL] GPU pipeline failed.
exit /b 1
