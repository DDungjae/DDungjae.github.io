@echo off
setlocal
rem tools\cv\cv.html -> assets\files\Jaewon_Seo_CV.pdf (Chrome/Edge headless print)
cd /d "%~dp0"

set "BROWSER="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER (
  echo  [X] Chrome or Edge was not found.
  exit /b 1
)

set "SRC=%~dp0cv.html"
set "OUT=%~dp0..\..\assets\files\Jaewon_Seo_CV.pdf"

"%BROWSER%" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="%OUT%" "file:///%SRC:\=/%" 2>nul
if errorlevel 1 (
  echo  [X] PDF build failed.
  exit /b 1
)
echo  [OK] Wrote assets\files\Jaewon_Seo_CV.pdf
