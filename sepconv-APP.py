import torch
import cupy
import pickle
import platform
import ctypes
import threading
import tkinter.font as tkFont
import multiprocessing
from tkinter.filedialog import askopenfilename,askdirectory
from tkinter import ttk
import tkinter.messagebox
import tkinter
import ffmpeg
import getopt
import math
import numpy
import time
import os
import re
import PIL
import PIL.Image
import random
import shutil
import sys
import tempfile

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) #要求pyTorch版本>=1.3.0
torch.set_grad_enabled(False) #确保不计算梯度来提高运行效率
torch.backends.cudnn.enabled = True #确保使用cudnn来提高计算性能

arguments_strModel = '' #选择用哪个模型l1/lf
arguments_strPadding = '' #选择模型的处理方式paper/improved

__VERSION__ = 'beta0.11'

kernel_Sepconv_updateOutput = '''
	extern "C" __global__ void kernel_Sepconv_updateOutput(
		const int n,
		const float* input,
		const float* vertical,
		const float* horizontal,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float fltOutput = 0.0;

		const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX = ( intIndex                                                    ) % SIZE_3(output);

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
			for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
				fltOutput += VALUE_4(input, intN, intC, intY + intFilterY, intX + intFilterX) * VALUE_4(vertical, intN, intFilterY, intY, intX) * VALUE_4(horizontal, intN, intFilterX, intY, intX);
			}
		}

		output[intIndex] = fltOutput;
	} }
'''

def cupy_kernel(strFunction, objVariables):
	strKernel = globals()[strFunction]

	while True:
		objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objMatch is None:
			break

		intArg = int(objMatch.group(2))

		strTensor = objMatch.group(4)
		intSizes = objVariables[strTensor].size()

		strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))

	while True:
		objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objMatch is None:
			break

		intArgs = int(objMatch.group(2))
		strArgs = objMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')

	return strKernel

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)

class _FunctionSepconv(torch.autograd.Function):
	@staticmethod
	def forward(self, input, vertical, horizontal):
		self.save_for_backward(input, vertical, horizontal)

		intSample = input.shape[0]
		intInputDepth = input.shape[1]
		intInputHeight = input.shape[2]
		intInputWidth = input.shape[3]
		intFilterSize = min(vertical.shape[1], horizontal.shape[1])
		intOutputHeight = min(vertical.shape[2], horizontal.shape[2])
		intOutputWidth = min(vertical.shape[3], horizontal.shape[3])

		assert(intInputHeight - intFilterSize == intOutputHeight - 1)
		assert(intInputWidth - intFilterSize == intOutputWidth - 1)

		assert(input.is_contiguous() == True)
		assert(vertical.is_contiguous() == True)
		assert(horizontal.is_contiguous() == True)

		output = input.new_zeros([ intSample, intInputDepth, intOutputHeight, intOutputWidth ])

		if input.is_cuda == True:
			n = output.nelement()
			cupy_launch('kernel_Sepconv_updateOutput', cupy_kernel('kernel_Sepconv_updateOutput', {
				'input': input,
				'vertical': vertical,
				'horizontal': horizontal,
				'output': output
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), output.data_ptr() ]
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		return output

	@staticmethod
	def backward(self, gradOutput):
		input, vertical, horizontal = self.saved_tensors

		intSample = input.shape[0]
		intInputDepth = input.shape[1]
		intInputHeight = input.shape[2]
		intInputWidth = input.shape[3]
		intFilterSize = min(vertical.shape[1], horizontal.shape[1])
		intOutputHeight = min(vertical.shape[2], horizontal.shape[2])
		intOutputWidth = min(vertical.shape[3], horizontal.shape[3])

		assert(intInputHeight - intFilterSize == intOutputHeight - 1)
		assert(intInputWidth - intFilterSize == intOutputWidth - 1)

		assert(gradOutput.is_contiguous() == True)

		gradInput = input.new_zeros([ intSample, intInputDepth, intInputHeight, intInputWidth ]) if self.needs_input_grad[0] == True else None
		gradVertical = input.new_zeros([ intSample, intFilterSize, intOutputHeight, intOutputWidth ]) if self.needs_input_grad[1] == True else None
		gradHorizontal = input.new_zeros([ intSample, intFilterSize, intOutputHeight, intOutputWidth ]) if self.needs_input_grad[2] == True else None

		if input.is_cuda == True:
			raise NotImplementedError()

		elif input.is_cuda == False:
			raise NotImplementedError()

		return gradInput, gradVertical, gradHorizontal

def FunctionSepconv(tenInput, tenVertical, tenHorizontal):
	return _FunctionSepconv.apply(tenInput, tenVertical, tenHorizontal)

class ModuleSepconv(torch.nn.Module):
	def __init__(self):
		super(ModuleSepconv, self).__init__()

	def forward(self, tenInput, tenVertical, tenHorizontal):
		return _FunctionSepconv.apply(tenInput, tenVertical, tenHorizontal)

class Network(torch.nn.Module):
    def __init__(self, arguments_strModel):
        super(Network, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )

        self.netConv1 = Basic(6, 32)
        self.netConv2 = Basic(32, 64)
        self.netConv3 = Basic(64, 128)
        self.netConv4 = Basic(128, 256)
        self.netConv5 = Basic(256, 512)

        self.netDeconv5 = Basic(512, 512)
        self.netDeconv4 = Basic(512, 256)
        self.netDeconv3 = Basic(256, 128)
        self.netDeconv2 = Basic(128, 64)

        self.netUpsample5 = Upsample(512, 512)
        self.netUpsample4 = Upsample(256, 256)
        self.netUpsample3 = Upsample(128, 128)
        self.netUpsample2 = Upsample(64, 64)

        self.netVertical1 = Subnet()
        self.netVertical2 = Subnet()
        self.netHorizontal1 = Subnet()
        self.netHorizontal2 = Subnet()

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(os.path.join(os.getcwd(), 'network', 'network-' + arguments_strModel + '.pytorch')).items() })


    def forward(self, tenFirst, tenSecond):
        tenConv1 = self.netConv1(torch.cat([ tenFirst, tenSecond ], 1))
        tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=tenConv1, kernel_size=2, stride=2, count_include_pad=False))
        tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=tenConv2, kernel_size=2, stride=2, count_include_pad=False))
        tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=tenConv3, kernel_size=2, stride=2, count_include_pad=False))
        tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=tenConv4, kernel_size=2, stride=2, count_include_pad=False))

        tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.avg_pool2d(input=tenConv5, kernel_size=2, stride=2, count_include_pad=False)))
        tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
        tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
        tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))

        tenCombine = tenDeconv2 + tenConv2

        tenFirst = torch.nn.functional.pad(input=tenFirst, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')
        tenSecond = torch.nn.functional.pad(input=tenSecond, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')

        tenDot1 = FunctionSepconv(tenInput=tenFirst, tenVertical=self.netVertical1(tenCombine), tenHorizontal=self.netHorizontal1(tenCombine))
        tenDot2 = FunctionSepconv(tenInput=tenSecond, tenVertical=self.netVertical2(tenCombine), tenHorizontal=self.netHorizontal2(tenCombine))

        return tenDot1 + tenDot2

netNetwork = None
arguments_strModel = ''
arguments_strPadding = ''

def estimate(tenFirst, tenSecond, arguments_strModel, arguments_strPadding):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network(arguments_strModel).cuda().eval()

    assert(tenFirst.shape[1] == tenSecond.shape[1])
    assert(tenFirst.shape[2] == tenSecond.shape[2])

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    if arguments_strPadding == 'paper':
        intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ,int(math.floor(51 / 2.0))

    elif arguments_strPadding == 'improved':
        intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = 0, 0, 0, 0

    intPreprocessedWidth = intPaddingLeft + intWidth + intPaddingRight
    intPreprocessedHeight = intPaddingTop + intHeight + intPaddingBottom

    if intPreprocessedWidth != ((intPreprocessedWidth >> 7) << 7):
        intPreprocessedWidth = (((intPreprocessedWidth >> 7) + 1) << 7)
    
    if intPreprocessedHeight != ((intPreprocessedHeight >> 7) << 7):
        intPreprocessedHeight = (((intPreprocessedHeight >> 7) + 1) << 7)

    intPaddingRight = intPreprocessedWidth - intWidth - intPaddingLeft
    intPaddingBottom = intPreprocessedHeight - intHeight - intPaddingTop

    tenPreprocessedFirst = torch.nn.functional.pad(input=tenPreprocessedFirst, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')
    tenPreprocessedSecond = torch.nn.functional.pad(input=tenPreprocessedSecond, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')

    return torch.nn.functional.pad(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), pad=[ 0 - intPaddingLeft, 0 - intPaddingRight, 0 - intPaddingTop, 0 - intPaddingBottom ], mode='replicate')[0, :, :, :].cpu()

def genrate(firstImage,secondImage,outputImage,arguments_strModel,arguments_strPadding):
    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(firstImage))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(secondImage))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenOutput = estimate(tenFirst, tenSecond, arguments_strModel, arguments_strPadding)
    PIL.Image.fromarray((tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(outputImage)

def getFrameRate(target):
    info = ffmpeg.probe(target)
    vs = next(c for c in info['streams'] if c['codec_type'] == 'video')
    fps = int(vs['r_frame_rate'].split('/')[0])/int(vs['r_frame_rate'].split('/')[1])
    return fps

input_file_path = ''
output_floder_path = ''
fps = 0
target_fps = 0
moudle_chose = '2'
padding_chose = '2'
multiple_chose = '1'
cut_fps_chose = False

def moudle_chose_func(event):
    global moudle_chose
    if moudle_chose_combobox.get() == 'lf':
        moudle_chose = '2'
    else:
        moudle_chose = '1'

def padding_chose_func(event):
    global padding_chose
    if padding_chose_combobox.get() == 'improved':
        padding_chose = '2'
    else:
        padding_chose = '1'

def multiple_chose_func(event):
    global multiple_chose,target_fps,fps
    if multiple_chose_combobox.get() == '2x':
        multiple_chose = '1'
        target_fps = fps*2
    elif multiple_chose_combobox.get() == '4x':
        multiple_chose = '2'
        target_fps = fps*4
    else:
        multiple_chose = '3'
        target_fps = fps*8

    if target_fps != 0:
        output_fps_label['text'] = '输出帧率为：'+str(target_fps)+'fps'

def cut_fps_chose_func():
    global cut_fps_chose
    if cut_fps_var.get() == 1:
        cut_fps_chose = True
    else:
        cut_fps_chose = False

def cut_fps_entry_only_num(content):
    try:
        float(content)
    except:
        if content != '':
            return False
    return True

def input_file_func():
    global output_floder_path,input_file_path,fps,target_fps,multiple_chose
    input_file_path = askopenfilename(title="请选择一个要打开的视频文件",filetypes=[("任何支持格式的视频文件", "*.mp4;*.mkv;*.flv;*.avi;*.mov;*.rmvb")])
    input_file_path_label['text'] = input_file_path
    fps = getFrameRate(input_file_path)
    input_fps_label['text'] = '输入帧率为：'+str(fps)+'fps'
    if multiple_chose == '1':
        target_fps = fps*2
    elif add_type == '2':
        target_fps = fps*4
    else:
        target_fps = fps*8
    output_fps_label['text'] = '输出帧率为：'+str(target_fps)+'fps'

def output_floder_func():
    global output_floder_path
    output_floder_path = askdirectory(title="请选择输出文件夹")
    output_floder_path_label['text'] = output_floder_path

def render_background_func(q,qp,input_file_path,output_floder_path,moudle_chose,cut_fps_chose,cut_fps_text,target_fps,arguments_strPadding,arguments_strModel,multiple_chose):
    def set_process_bar_maxium(num):
        q.put('process_bar_maxium')
        q.put(num)
    def set_remain_time_text(text):
        q.put('set_remain_time_text')
        q.put(text)
    def set_process_persent_text(text):
        q.put('set_process_persent_text')
        q.put(text)
    def set_process_bar_value(num):
        q.put('set_process_bar_value')
        q.put(num)
    def set_status_bar_text(text):
        q.put('set_status_bar_text')
        q.put(text)
    def get_pause_or_stop_status():
        qp.put('get_status')
        status = qp.get(block=True)
        return status
    def render_finish():
        q.put('finish')
        q.put('finish')
    def start_or_save():
        data = qp.get(block=True)
        return data
    file_name = '.'.join(os.path.basename(input_file_path).split('.')[:-1])
    original_frames_path = os.path.join(output_floder_path,'original_frames')
    interpolated_frames_path = os.path.join(output_floder_path,'interpolated_frames')
    output_videos_path = os.path.join(output_floder_path,'output_videos')
    temp_audio_path = os.path.join(output_floder_path,'audio_temp')

    print(original_frames_path)
    
    os.makedirs(original_frames_path)
    os.makedirs(temp_audio_path)
    
    if moudle_chose == '1':
        arguments_strModel = 'l1'
    else:
        arguments_strModel = 'lf'

    if padding_chose == '1':
        arguments_strPadding = 'paper'
    else:
        arguments_strPadding = 'improved'

    if cut_fps_chose:
        output_fps = float(cut_fps_text)

    print('\n正在提取视频帧...')
    set_status_bar_text('正在提取视频帧...')
    os.system('ffmpeg -i \"'+input_file_path+'\" \"'+os.path.join(original_frames_path,'%09d.png\"'))
    print('提取完毕\n')
    set_status_bar_text('提取完毕')

    print('正在提取音频...')
    set_status_bar_text('正在提取音频...')
    os.system('ffmpeg -i \"'+input_file_path+'\" -vn \"'+os.path.join(temp_audio_path,file_name+'.mp3\"'))
    print('提取完毕\n')
    set_status_bar_text('提取完毕')

    frame_num = len([lists for lists in os.listdir(original_frames_path) if os.path.isfile(os.path.join(original_frames_path,lists))])
    set_process_bar_maxium(frame_num)
    print('一共有'+str(frame_num)+'帧需要处理\n')
    set_status_bar_text('一共有'+str(frame_num)+'帧需要处理')

    print('开始处理...\n')
    set_status_bar_text('开始处理...')

    os.makedirs(interpolated_frames_path)
    shutil.copyfile(os.path.join(original_frames_path,str(1).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(1).zfill(9)+'.png'))

    t1 = 0
    t2 = 0
    t_all = 0
    output_frame_counter = 1

    start_frame = 1

    stop = False
    
    for i in range(start_frame,frame_num):
        now_status = get_pause_or_stop_status()
        if now_status != 'keep':
            if now_status == 'pause':
                print('暂停中')
                set_status_bar_text('暂停中')
                while True:
                    data = start_or_save()
                    if data == 'start':
                        break
            elif data == 'save':
                save_data = [input_file_path,output_floder_path,moudle_chose,cut_fps_chose,cut_fps_text,target_fps,arguments_strPadding,arguments_strModel,multiple_chose,t_all,i]
                f = open(os.path.join(output_floder_path,'save.sepconv'),'wb')
                pickle.dump(save_data,f,2)
                f.close()

        if stop:
            break
        
        if t1 == 0:
            print('正在处理'+str(i)+'/'+str(frame_num)+'帧,完成了'+str(round((i-1)/frame_num*100,3))+'%,预计剩余时间未知')
            set_status_bar_text('正在处理'+str(i)+'/'+str(frame_num)+'帧,完成了'+str(round((i-1)/frame_num*100,3))+'%,预计剩余时间未知')
            set_remain_time_text('预计剩余时间：未知')
            set_process_persent_text(str(round((i-1)/frame_num*100,3))+'%')
            set_process_bar_value(i)
        else:
            print('正在处理'+str(i)+'/'+str(frame_num)+'帧,完成了'+str(round((i-1)/frame_num*100,3))+'%,预计剩余时间'+str(round(t_all/(i-1)*(frame_num-i),1))+'s')
            set_status_bar_text('正在处理'+str(i)+'/'+str(frame_num)+'帧,完成了'+str(round((i-1)/frame_num*100,3))+'%,预计剩余时间'+str(round(t_all/(i-1)*(frame_num-i),1))+'s')
            set_remain_time_text('预计剩余时间：'+str(round(t_all/(i-1)*(frame_num-i),1))+'s')
            set_process_persent_text(str(round((i-1)/frame_num*100,3))+'%')
            set_process_bar_value(i)
            
        if multiple_chose == '1': #2x
            t1 = time.time()
            shutil.copyfile(os.path.join(original_frames_path,str(i+1).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'))
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+1).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            output_frame_counter = output_frame_counter+2
            t2 = time.time()
        elif multiple_chose == '2': #4x
            t1 = time.time()
            shutil.copyfile(os.path.join(original_frames_path,str(i+1).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'))
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+1).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+3).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            output_frame_counter = output_frame_counter+4
            t2 = time.time()
        elif multiple_chose == '3': #8x
            t1 = time.time()
            shutil.copyfile(os.path.join(original_frames_path,str(i+1).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+8).zfill(9)+'.png'))
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+8).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+8).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+6).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+1).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter+2).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+3).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter+4).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+6).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+5).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            genrate(os.path.join(interpolated_frames_path,str(output_frame_counter+6).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+8).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter+7).zfill(9)+'.png'),arguments_strModel,arguments_strPadding)
            output_frame_counter = output_frame_counter+8
            t2 = time.time()
        t_all = t_all+t2-t1

    print('正在处理'+str(frame_num)+'/'+str(frame_num)+'帧,完成了'+str(round((frame_num-1)/frame_num*100,3))+'%,预计剩余时间0s')
    set_status_bar_text('正在处理'+str(frame_num)+'/'+str(frame_num)+'帧,完成了'+str(round((frame_num-1)/frame_num*100,3))+'%,预计剩余时间0s')
    set_remain_time_text('预计剩余时间：0s')
    set_process_persent_text(str(round((frame_num-1)/frame_num*100,3))+'%')
    set_process_bar_value(frame_num)
    if multiple_chose == '1': #2x
        for i in range(0,2):
            shutil.copyfile(os.path.join(original_frames_path,str(frame_num).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'))
            output_frame_counter = output_frame_counter+1
    elif multiple_chose == '2': #4x
        for i in range(0,4):
            shutil.copyfile(os.path.join(original_frames_path,str(frame_num).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'))
            output_frame_counter = output_frame_counter+1
    elif multiple_chose == '3': #8x
        for i in range(0,8):
            shutil.copyfile(os.path.join(original_frames_path,str(frame_num).zfill(9)+'.png'),os.path.join(interpolated_frames_path,str(output_frame_counter).zfill(9)+'.png'))
            output_frame_counter = output_frame_counter+1
    
    print('处理完成\n')
    set_status_bar_text('处理完成')
    set_process_persent_text('100%')
    print('正在合成视频...')
    set_status_bar_text('正在合成视频...')
    os.makedirs(output_videos_path)
    
    os.system('ffmpeg -f image2 -r '+str(target_fps)+' -i \"'+os.path.join(interpolated_frames_path,'%09d.png')+'\" -i \"'+os.path.join(temp_audio_path,file_name+'.mp3')+'\" -vcodec h264 -acodec aac \"'+os.path.join(output_videos_path,str(target_fps)+'fps_'+file_name+'.mp4\"'))

    print('视频合成完毕\n')
    set_status_bar_text('视频合成完毕')

    if cut_fps_chose:
        print('正在降低帧率...')
        set_status_bar_text('正在降低帧率...')
        os.system('ffmpeg -i \"'+os.path.join(output_videos_path,str(target_fps)+'fps_'+file_name+'.mp4')+'\" -r '+str(output_fps)+' \"'+os.path.join(output_videos_path,str(output_fps)+'fps_'+file_name+'.mp4\"'))
        print('降低完成\n')
        set_status_bar_text('降低完成')

    print('处理完成\n')
    set_status_bar_text('处理完成')
    render_finish()

def render_pause_control_func(qp):
    global render_shoud_pause,render_should_save,render_should_stop
    while True:
        data = qp.get(block=True)
        if render_should_pause:
            qp.put('pause')
        elif render_should_stop:
            qp.put('stop')
        else:
            qp.put('keep')
        if render_should_pause:
            while True:
                if not render_should_pause:
                    qp.put('start')
                    break
                elif render_should_save:
                    render_should_save = False
                    qp.put('save')
                elif render_should_stop:
                    render_should_stop = False
                    qp.put('stop')

def render_communicate_func():
    global input_file_path,output_floder_path,moudle_chose,cut_fps_chose,target_fps,arguments_strPadding,arguments_strModel,is_rendering
    cut_fps_text = cut_fps_input.get()
    q = multiprocessing.Queue()
    qp = multiprocessing.Queue()
    p = multiprocessing.Process(target=render_background_func,args=(q,qp,input_file_path,output_floder_path,moudle_chose,cut_fps_chose,target_fps,cut_fps_text,arguments_strPadding,arguments_strModel,multiple_chose))
    p.start()
    t = threading.Thread(target=render_pause_control_func,args=(qp,))
    t.start()
    while True:
        data = q.get(block=True)
        data2 = q.get(block=True)
        if data == 'process_bar_maxium':
            process_bar['maximum'] = data2
        elif data == 'set_remain_time_text':
            remain_time['text'] = data2
        elif data == 'set_process_persent_text':
            process_persent['text'] = data2
        elif data == 'set_process_bar_value':
            process_bar['value'] = data2
        elif data == 'set_process_bar_value':
            text = cut_fps_input.get()
            q.put(text)
        elif data == 'set_status_bar_text':
            status_bar['text'] = data2
        elif data == 'finish':
            break
    is_rendering = False

is_rendering = False
render_should_pause = False
render_should_save = False
render_should_stop = False

def render_bootloader_func():
    global is_rendering,render_should_pause
    if not is_rendering:
        t = threading.Thread(target=render_communicate_func)
        t.start()
        is_rendering = True
    if render_should_pause == True:
        render_should_pause = False

def render_pause():
    global render_should_pause
    render_should_pause = True
    status_bar['text'] = '正在尝试暂停...'

def render_stop():
    global render_should_stop
    render_should_stop = True
    status_bar['text'] = '正在尝试停止...'

def render_save():
    global render_should_save
    render_should_save = True

def render_load():
    if tkinter.messagebox.askyesno('加载', '确认要加载吗？当前未保存数据将丢失'):
        load_file_path = askopenfilename(title="请选择sepconv-APP的保存文件，会保存在目标文件夹下",filetypes=[("sepconv-APP保存文件", "save.sepconv")])
        f = open(load_file_path,'rb')
        load_data = pickle,load(f)
        f.close()
        print(load_data)
    else:
        pass

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('----------sepconv-APP '+__VERSION__+'----------\n')
    root = tkinter.Tk()

    if(platform.system()=='Windows'): #Windows专用对付dpi模糊的部分
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        ScaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0)
        root.tk.call('tk','scaling',ScaleFactor/75)
    
    root.title('sepconv-APP '+__VERSION__)
    tkinter.Label(root,text='sepconv-APP '+__VERSION__,font=tkFont.Font(size=20,weight=tkFont.BOLD)).grid(row=0,column=0,columnspan=2)
    
    tkinter.Button(root,text='输入视频',width=20,heigh=1,command=input_file_func).grid(row=1,sticky='nsew',pady=3,padx=3)
    input_file_path_label = tkinter.Label(root,text='输入视频的目录',width=50,heigh=1,anchor='w')
    input_file_path_label.grid(row=1,column=1,sticky='nsew',pady=3,padx=3)

    input_fps_label = tkinter.Label(root,text='输入帧率为：未知',width=50,heigh=1,anchor='n')
    input_fps_label.grid(row=2,column=0,sticky='nsew',pady=3,padx=3)
    
    tkinter.Button(root,text='输出视频',width=20,heigh=1,command=output_floder_func).grid(row=3,sticky='nsew',pady=3,padx=3)
    output_floder_path_label = tkinter.Label(root,text='输出视频的目录',width=50,heigh=1,anchor='w')
    output_floder_path_label.grid(row=3,column=1,sticky='nsew',pady=3,padx=3)

    tkinter.Frame(height=2,bd=1,relief=tkinter.GROOVE,padx=10,width=950).grid(row=4,column=0,columnspan=2,padx=10,pady=6)

    tkinter.Label(root,text='要使用的模型',width=50,heigh=1,anchor='n').grid(row=5,column=0,sticky='nsew',pady=3,padx=3)
    moudle_chose_combobox = tkinter.ttk.Combobox(root,width=50,heigh=1,state='readonly',values=('l1','lf'))
    moudle_chose_combobox.current(1)
    moudle_chose_combobox.grid(row=5,column=1,sticky='nsew',pady=3,padx=3)
    moudle_chose_combobox.bind("<<ComboboxSelected>>",moudle_chose_func)

    tkinter.Label(root,text='模型的处理方式',width=50,heigh=1,anchor='n').grid(row=6,column=0,sticky='nsew',pady=3,padx=3)
    padding_chose_combobox = tkinter.ttk.Combobox(root,width=50,heigh=1,state='readonly',values=('paper','improved'))
    padding_chose_combobox.current(1)
    padding_chose_combobox.grid(row=6,column=1,sticky='nsew',pady=3,padx=3)
    padding_chose_combobox.bind("<<ComboboxSelected>>",padding_chose_func)

    tkinter.Frame(root,height=2,bd=1,relief=tkinter.GROOVE,padx=10,width=950).grid(row=7,column=0,columnspan=2,padx=10,pady=6)

    tkinter.Label(root,text='要补帧的倍数',width=50,heigh=1,anchor='n').grid(row=8,column=0,sticky='nsew',pady=3,padx=3)
    multiple_chose_combobox = tkinter.ttk.Combobox(root,width=50,heigh=1,state='readonly',values=('2x','4x','8x'))
    multiple_chose_combobox.current(0)
    multiple_chose_combobox.grid(row=8,column=1,sticky='nsew',pady=3,padx=3)
    multiple_chose_combobox.bind("<<ComboboxSelected>>",multiple_chose_func)

    output_fps_label = tkinter.Label(root,text='输出帧率为：未知',width=50,heigh=1,anchor='n')
    output_fps_label.grid(row=9,column=0,sticky='nsew',pady=3,padx=3)

    tkinter.Frame(root,height=2,bd=1,relief=tkinter.GROOVE,padx=10,width=950).grid(row=10,column=0,columnspan=2,padx=10,pady=6)

    frame1 = tkinter.Frame(root)
    frame1.grid(row=11,column=0,columnspan=2,sticky='nsew',pady=3,padx=3)
    cut_fps_var = tkinter.IntVar()
    tkinter.Checkbutton(frame1,text='是否要降低输出帧率',variable=cut_fps_var,command=cut_fps_chose_func).grid(row=0,column=0,sticky='nsew',pady=3,padx=3)
    tkinter.Label(frame1,text='要降低到的帧率：',heigh=1,anchor='e').grid(row=0,column=1,sticky='nsew',pady=3,padx=3)
    cut_fps_value = tkinter.IntVar()
    cut_fps_value.set('60')
    cut_fps_entry_only_num_tk = root.register(cut_fps_entry_only_num)
    cut_fps_input = tkinter.Entry(frame1,width=80,validate='key',textvariable=cut_fps_value,validatecommand=(cut_fps_entry_only_num_tk, '%P'))
    cut_fps_input.grid(row=0,column=2,sticky='nsew',pady=3,padx=3)

    frame2 = tkinter.Frame(root)
    frame2.grid(row=12,column=0,columnspan=2,sticky='nsew',pady=3,padx=3)
    remain_time = tkinter.Label(frame2,text='预计剩余时间：未知',heigh=1,anchor='w')
    remain_time.grid(row=0,column=0,sticky='nsew',pady=3,padx=3)
    process_persent = tkinter.Label(frame2,text='0%',heigh=1,anchor='w')
    process_persent.grid(row=0,column=1,sticky='nsew',pady=3,padx=3)
    process_bar = ttk.Progressbar(frame2,length=800,mode="determinate",orient=tkinter.HORIZONTAL)
    process_bar.grid(row=0,column=2,sticky='nsew',pady=3,padx=3)

    frame3 = tkinter.Frame(root)
    frame3.grid(row=13,column=0,columnspan=3,sticky='nsew',pady=3,padx=3)
    tkinter.Button(frame3,text='开始渲染',width=57,heigh=1,command=render_bootloader_func).grid(row=0,column=0,sticky='nsew',pady=3,padx=3)
    tkinter.Button(frame3,text='暂停渲染',width=57,heigh=1,command=render_pause).grid(row=0,column=1,sticky='nsew',pady=3,padx=3)
    #tkinter.Button(frame3,text='保存进度',width=27,heigh=1,command=render_save).grid(row=0,column=2,sticky='nsew',pady=3,padx=3)
    #tkinter.Button(frame3,text='加载进度',width=27,heigh=1,command=render_load).grid(row=0,column=3,sticky='nsew',pady=3,padx=3)

    status_bar = tkinter.Label(root,text="启动完成",bd=1,relief=tkinter.SUNKEN,anchor='w')
    status_bar.grid(row=15,column=0,columnspan=2,sticky='we')
    
    root.mainloop()
