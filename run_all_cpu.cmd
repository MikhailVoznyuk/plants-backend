@echo off
setlocal
echo === Building CPU image ===
docker build -t treehealth-cpu -f Dockerfile.cpu . || goto :err
echo === Starting CPU compose ===
docker compose -f docker-compose.cpu.yml up -d --build || goto :err
echo === Wait 5s for warmup ===
ping -n 6 127.0.0.1 >NUL
echo === Running suite ===
call tools\run_suite.cmd || goto :err
echo [OK] CPU pipeline passed.
exit /b 0
:err
echo [FAIL] CPU pipeline failed.
exit /b 1
