@echo off
REM Sundew SDK Quick Test for Surface
REM Run this to validate SDK installation

echo ========================================
echo Sundew SDK - Surface Testing
echo ========================================
echo.

REM Activate virtual environment
echo [1/5] Activating virtual environment...
call .venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Virtual environment not found. Run: python -m venv .venv
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Install dependencies
echo [2/5] Installing SDK dependencies...
pip install -q grpcio grpcio-tools protobuf pytest >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Generate IPC bindings
echo [3/5] Generating IPC bindings...
python tools\generate_ipc_bindings.py >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to generate IPC bindings
    pause
    exit /b 1
)
echo ✓ IPC bindings generated
echo.

REM Run IPC demo
echo [4/5] Running IPC demo...
echo.
python examples\ipc_demo.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: IPC demo failed
    pause
    exit /b 1
)
echo.
echo ✓ IPC demo completed
echo.

REM Run test suite
echo [5/5] Running SDK test suite...
echo.
pytest tests\test_ipc*.py tests\test_grpc*.py -v
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Tests failed
    pause
    exit /b 1
)
echo.

echo ========================================
echo ✓ All Surface tests PASSED!
echo ========================================
echo.
echo SDK is ready for deployment.
echo.
echo Next steps:
echo   1. Test two-Surface network setup (see docs\SURFACE_TESTING_GUIDE.md)
echo   2. Try Google Colab testing (see docs\COLAB_TESTING_GUIDE.md)
echo   3. Deploy to Raspberry Pi when ready
echo.
pause
