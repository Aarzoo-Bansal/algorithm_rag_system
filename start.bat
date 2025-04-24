@echo off
echo Starting rag_service.py...
start /B python api/rag_service.py

REM Wait for the server to be ready (port 8000)
:waitloop
timeout /T 1 >nul
powershell -Command "try { (New-Object Net.Sockets.TcpClient).Connect('localhost', 8000); exit 0 } catch { exit 1 }"
if errorlevel 1 goto waitloop

echo rag_service.py is up. Starting app.py...
start /B python app.py

REM Wait for both processes if needed
pause
