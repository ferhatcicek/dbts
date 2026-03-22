@echo off
setlocal enabledelayedexpansion

echo ================================================
echo  Deformasyonel Brakisefali Tespit Sistemi
echo  Sanal Ortam Kurulum Scripti
echo ================================================
echo.

:: Python kontrolu
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi! Lutfen Python 3.9+ yukleyin.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python bulundu:
python --version
echo.

:: Mevcut venv varsa sil
if exist "venv" (
    echo [BILGI] Mevcut venv siliniyor...
    rmdir /s /q venv
)

:: Yeni venv olustur
echo [BILGI] Sanal ortam olusturuluyor...
python -m venv venv
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam olusturulamadi!
    pause
    exit /b 1
)
echo [OK] Sanal ortam olusturuldu.
echo.

:: Activate et
echo [BILGI] Sanal ortam aktive ediliyor...
call venv\Scripts\activate.bat

:: pip guncelle
echo [BILGI] pip guncelleniyor...
python -m pip install --upgrade pip --quiet
echo [OK] pip guncellendi.
echo.

:: Kutuphaneleri yukle
echo [BILGI] Kutuphaneler yukleniyor (bu birkaç dakika surebilir)...
echo.

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [HATA] Kutuphaneler yuklenemedi!
    echo Lütfen requirements.txt dosyasini kontrol edin.
    pause
    exit /b 1
)

echo.
echo ================================================
echo  Kurulum tamamlandi!
echo  
echo  Uygulamayi baslatmak icin:
echo    1. call venv\Scripts\activate.bat
echo    2. python main.py
echo  
echo  veya dogrudan:
echo    run.bat
echo ================================================
echo.

:: run.bat olustur
echo @echo off > run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo python main.py >> run.bat
echo pause >> run.bat

echo [OK] run.bat olusturuldu.
pause
