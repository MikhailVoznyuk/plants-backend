@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Prereqs for local tester
py -m pip install -U pillow requests >NUL 2>&1

REM Generate synthetic tests
py tools\gen_tests.py

REM Make output dir
mkdir test_out 2>NUL

REM Iterate images and call the running service at localhost:8000
for %%F in (tests\*.jpg) do (
  echo Sending %%F
  curl -s -X POST "http://localhost:8000/infer" -F "file=@%%F" > "test_out\%%~nF.json"
  IF ERRORLEVEL 1 (
     echo ^> Request failed for %%F
     exit /b 1
  )
  REM Extract report_json_path from response (very crude) and validate
  for /f "usebackq tokens=1,* delims=:" %%a in (`findstr /i "report_json_path" "test_out\%%~nF.json"`) do (
    set LINE=%%b
  )
  set LINE=%LINE: =%
  set LINE=%LINE:\=/%
  set LINE=%LINE:,"=%
  set LINE=%LINE:,=%
  set REPORT=%LINE%

  REM If the service wrote absolute container path, we can't read it here.
  REM So just validate structure of the response JSON instead:
  py tools\validate_report.py "test_out\%%~nF.json"
  IF ERRORLEVEL 1 exit /b 1
)

echo.
echo [OK] Test suite finished successfully.
