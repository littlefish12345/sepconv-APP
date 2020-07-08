@echo off

md temp-build-onefile
md build-onefile
md .\build-onefile\sepconv-APP-onefile
copy sepconv-APP.py .\temp-build-onefile
copy sepconv-APP-onefile.spec .\temp-build-onefile
cd temp-build-onefile

pyinstaller sepconv-APP-onefile.spec

cd ..
copy temp-build-onefile\dist\sepconv-APP.exe .\build-onefile\sepconv-APP-onefile
xcopy network .\build-onefile\sepconv-APP-onefile\network\ /e
copy ffmpeg.exe .\build-onefile\sepconv-APP-onefile
copy ffprobe.exe .\build-onefile\sepconv-APP-onefile

rd /s /q temp-build-onefile

pause