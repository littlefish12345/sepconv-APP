@echo off

md temp-build
md build
md .\build\sepconv-APP
copy sepconv-APP.py .\temp-build
copy sepconv-APP.spec .\temp-build
cd temp-build

pyinstaller sepconv-APP.spec

cd ..
xcopy temp-build\dist\sepconv-APP\* .\build\sepconv-APP /e
md .\build\sepconv-APP\temp
xcopy network .\build\sepconv-APP\network\ /e
copy ffmpeg.exe .\build\sepconv-APP
copy ffprobe.exe .\build\sepconv-APP

rd /s /q temp-build

pause