@echo off
:: Check if the virtual environment exists
IF NOT EXIST "venv\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Install dependencies from requirements.txt if they are not installed
echo Installing dependencies...
pip install -r requirements.txt

:: Run the Dash application
echo Starting the application...
python dashapp.py

pause
