# sepconv-APP

## 简介
一个基于sepconv-slome的视频补帧软件

## 测试环境
### 系统环境
```
Windows 10 专业版 64bit 1909(仅在此版本测试过,但其他版本应该也是可行的)
python 3.6.4(仅在此版本测试过,但其他版本应该也是可行的)
CUDA 10.1(仅在此版本测试过,但其他版本应该也是可行的)
cuDNN 7.6.4
```

### 硬件环境
```
CPU: i7 8700k@3.7GHz
GPU: RTX 2080ti
RAM: 32G ddr4 3200MHz
```

### python依赖
```
cupy>=7.4.0
torch>=1.4.0
ffmpeg-python>=0.2.0
numpy>=1.18.1
pillow>=7.0.0
pyinstaller>=3.6
```

## 使用方法
### Windows 64bit
```
安装python3.6.4
安装cuda10.1
运行windows-static-libraries-download.bat
运行lib-install-cuda101.bat
完成
```

## 编译方法
### Windows 64bit
```
运行build-clear.bat
运行build.bat
```