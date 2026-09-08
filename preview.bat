@echo off
setlocal
cd /d "%~dp0"
title Local preview - DDungjae.github.io

where ruby >nul 2>nul
if errorlevel 1 (
  echo  [X] Ruby was not found. Install "Ruby+Devkit (x64)" from
  echo      https://rubyinstaller.org/downloads/  then reopen this window.
  pause
  exit /b 1
)

rem Ruby 가 GitHub 에서 테마를 내려받을 때 인증서 오류가 나는 것을 막기 위해
rem Git 이 갖고 있는 CA 번들을 씁니다. (Git 이 없으면 그냥 넘어갑니다)
if exist "%ProgramFiles%\Git\mingw64\etc\ssl\certs\ca-bundle.crt" (
  set "SSL_CERT_FILE=%ProgramFiles%\Git\mingw64\etc\ssl\certs\ca-bundle.crt"
)

ruby -v
call bundle config set --local path vendor/bundle >nul 2>nul
call bundle check >nul 2>nul
if errorlevel 1 (
  echo  First run: downloading gems. This takes a few minutes, only once.
  call bundle install
  if errorlevel 1 goto fail
)

echo.
echo  Site  : http://localhost:4000
echo  Stop  : Ctrl+C, then Y
echo  Note  : _config.yml 을 고쳤을 때만 이 창을 껐다 다시 켜세요.
echo.
call bundle exec jekyll serve --livereload --open-url
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo  Something failed. Copy the text above and paste it to Claude.
pause
exit /b 1
