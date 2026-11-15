@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"
if /I not "%MODE%"=="full" if /I not "%MODE%"=="smoke" (
  echo Usage: run_all.bat [full^|smoke]
  exit /b 2
)

set "REPO_WIN=%~dp0."
for /f "usebackq delims=" %%I in (`wsl.exe -- wslpath -a "%REPO_WIN%"`) do set "REPO_WSL=%%I"
if not defined REPO_WSL (
  echo Failed to map the repository into WSL.
  exit /b 3
)

wsl.exe -- bash -lc "cd '%REPO_WSL%' && if [ -x ~/miniconda3/bin/conda ]; then ~/miniconda3/bin/conda run --no-capture-output -n iml-research python -m scripts.reproduce --mode %MODE%; elif [ -x ~/anaconda3/bin/conda ]; then ~/anaconda3/bin/conda run --no-capture-output -n iml-research python -m scripts.reproduce --mode %MODE%; else echo 'Project Conda installation not found.' >&2; exit 4; fi"
if errorlevel 1 exit /b %errorlevel%

if /I "%MODE%"=="full" (
  set "MIKTEX_BIN=C:\Program Files\MiKTeX\miktex\bin\x64"
  if not exist "%MIKTEX_BIN%\latexmk.exe" (
    echo MiKTeX latexmk was not found. Results were generated. The paper was not rebuilt.
    exit /b 5
  )
  set "PATH=%MIKTEX_BIN%;%PATH%"
  pushd "%REPO_WIN%paper"
  "%MIKTEX_BIN%\latexmk.exe" -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build LoopLab.tex
  set "LATEX_STATUS=%errorlevel%"
  popd
  if not "!LATEX_STATUS!"=="0" exit /b !LATEX_STATUS!
)

echo LoopLab %MODE% run completed.
exit /b 0
