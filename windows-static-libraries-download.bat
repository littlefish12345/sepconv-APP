@echo off

del /F /S /Q ffmpeg.exe
del /F /S /Q ffprobe.exe
rd /q /s network

echo ��������ffmpeg...
wget https://github.com/littlefish12345/static-libraries/releases/download/ffmpeg-windows/ffmpeg.exe
wget https://github.com/littlefish12345/static-libraries/releases/download/ffmpeg-windows/ffprobe.exe

echo �������

echo ��������������...
mkdir network
wget https://github.com/littlefish12345/static-libraries/releases/download/sepconv-APP-networks/network-lf.pytorch -P ./network/network-lf.pytorch
wget https://github.com/littlefish12345/static-libraries/releases/download/sepconv-APP-networks/network-l1.pytorch -P ./network/network-l1.pytorch
echo �������

pause