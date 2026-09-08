@echo off
setlocal
cd /d "%~dp0"
title Notion import - DDungjae.github.io

rem 사용법: 노션 export zip 을 이 파일 위에 끌어다 놓거나,
rem         notion-import.bat "C:\...\Export-xxxx.zip" [post|project]

if "%~1"=="" (
  echo  Drag a Notion export .zip onto this file, or run:
  echo    notion-import.bat "path\to\Export-xxxx.zip" [post^|project]
  pause
  exit /b 1
)

set "KIND=%~2"
if "%KIND%"=="" set "KIND=post"

where python >nul 2>nul
if errorlevel 1 (
  echo  [X] Python was not found. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)

echo  ---- preview (nothing is written yet) ----
python tools\notion_to_jekyll.py "%~1" --type %KIND% --dry-run
echo.
set /p GO=" Write this file? [y/N] "
if /i not "%GO%"=="y" (
  echo  Cancelled. Tip: ask Claude to run /notion-import to adjust slug, tags, or excerpt.
  pause
  exit /b 0
)
python tools\notion_to_jekyll.py "%~1" --type %KIND%
echo.
echo  Done. Run preview.bat to check it, then commit and push.
pause
