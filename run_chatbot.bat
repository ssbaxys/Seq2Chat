@echo off
chcp 65001 > nul
title Генеративный чат-бот
echo ========================================
echo    Генеративный чат-бот (Seq2Seq)
echo ========================================
echo.

python --version > nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден!
    echo Установите Python с https://python.org
    pause
    exit /b 1
)

echo Проверка зависимостей...
pip show torch > nul 2>&1
if errorlevel 1 (
    echo Установка PyTorch...
    pip install torch
)

pip show numpy > nul 2>&1
if errorlevel 1 (
    echo Установка NumPy...
    pip install numpy
)

echo.
echo Запуск...
echo.
python chatbot.py

pause
