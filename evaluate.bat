@echo off
setlocal

if "%~1"=="" (
    echo Usage: evaluate.bat MODEL_NAME [EPISODES]
    exit /b 1
)

set "model=%~1"
set "episodes=%~2"
if "%episodes%"=="" set "episodes=50"

echo Running EmbodiedBAO full evaluation for model: %model% (episodes=%episodes%)
%ISAACSIM_ROOT%\python.bat main.py --model %model% --all-levels --episodes %episodes% --headless

endlocal
