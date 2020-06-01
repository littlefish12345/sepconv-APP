import torch
import cupy

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

__VERSION__ = 'beta0.1'

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
    def __init__(self):
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

def estimate(tenFirst, tenSecond):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()

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

def genrate(firstImage,secondImage,outputImage):
    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(firstImage))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(secondImage))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenOutput = estimate(tenFirst, tenSecond)
    PIL.Image.fromarray((tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(outputImage)

def getFrameRate(target):
    info = ffmpeg.probe(target)
    vs = next(c for c in info['streams'] if c['codec_type'] == 'video')
    fps = int(vs['r_frame_rate'].split('/')[0])/int(vs['r_frame_rate'].split('/')[1])
    return fps

def del_files(path):
    ls = os.listdir(path)
    for i in ls:
        os.remove(os.path.join(path,i))

if __name__ == '__main__':
    print('----------sepconv-APP '+__VERDION__+'----------\n')
    if not os.path.exists(os.path.join(os.getcwd(),'temp')):
        os.makedirs(os.path.join(os.getcwd(),'temp'))
    f = input('请输入要补帧的视频的路径：')
    fps = getFrameRate(f)
    print('这个视频的帧率是'+str(fps)+'fps\n')
    
    output_path = input('请输入输出文件夹的路径：')
    os.makedirs(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames'))
    print('')
    
    moudle_type = input('请选择要使用的模型(1:l1,2:lf)：')
    if moudle_type == '1':
        arguments_strModel = 'l1'
    else:
        arguments_strModel = 'lf'

    padding_type = input('请选择模型的处理方式(1:paper,2:improved)：')
    if moudle_type == '1':
        arguments_strPadding = 'paper'
    else:
        arguments_strPadding = 'improved'

    target_fps = 0
    
    add_type = input('请输入要补帧的倍数(1:2x,2:4x)：')
    if add_type == '1':
        target_fps = fps*2
    elif add_type == '2':
        target_fps = fps*4
        
    print('输出帧率将是'+str(target_fps)+'fps\n')
    
    print('正在提取视频帧...')
    os.system('ffmpeg -i '+f+' '+os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames','%09d.png'))
    print('提取完毕\n')

    frame_num = len([lists for lists in os.listdir(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames')) if os.path.isfile(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames',lists))])
    print('一共有'+str(frame_num)+'帧需要处理\n')

    print('开始处理...\n')

    os.makedirs(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames'))
    output_frame_counter = 1
    shutil.copyfile(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames',str(1).zfill(9)+'.png'),os.path.join(os.getcwd(),'temp','1.png'))

    t1 = 0
    t2 = 0
    t_all = 0
    
    for i in range(1,frame_num):
        if t1 == 0:
            print('正在处理'+str(i)+'/'+str(frame_num)+'帧,完成了'+str(round((i-1)/frame_num*100,3))+'%,预计剩余时间未知')
        else:
            print('正在处理'+str(i)+'/'+str(frame_num)+'帧,完成了'+str(round((i-1)/frame_num*100,3))+'%,预计剩余时间'+str(round(t_all/(i-1)*(frame_num-i),1))+'s')
            
        if add_type == '1': #x2
            t1 = time.time()
            shutil.copyfile(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames',str(i+1).zfill(9)+'.png'),os.path.join(os.getcwd(),'temp','3.png'))
            genrate(os.path.join(os.getcwd(),'temp','1.png'),os.path.join(os.getcwd(),'temp','3.png'),os.path.join(os.getcwd(),'temp','2.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','1.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter).zfill(9)+'.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','2.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter+1).zfill(9)+'.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','3.png'),os.path.join(os.getcwd(),'temp','1.png'))
            output_frame_counter = output_frame_counter+2
            t2 = time.time()
        elif add_type == '2': #x4
            t1 = time.time()
            shutil.copyfile(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames',str(i+1).zfill(9)+'.png'),os.path.join(os.getcwd(),'temp','5.png'))
            genrate(os.path.join(os.getcwd(),'temp','1.png'),os.path.join(os.getcwd(),'temp','5.png'),os.path.join(os.getcwd(),'temp','3.png'))
            genrate(os.path.join(os.getcwd(),'temp','1.png'),os.path.join(os.getcwd(),'temp','3.png'),os.path.join(os.getcwd(),'temp','2.png'))
            genrate(os.path.join(os.getcwd(),'temp','3.png'),os.path.join(os.getcwd(),'temp','5.png'),os.path.join(os.getcwd(),'temp','4.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','1.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter).zfill(9)+'.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','2.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter+1).zfill(9)+'.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','3.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter+2).zfill(9)+'.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','4.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter+3).zfill(9)+'.png'))
            shutil.move(os.path.join(os.getcwd(),'temp','5.png'),os.path.join(os.getcwd(),'temp','1.png'))
            output_frame_counter = output_frame_counter+4
            t2 = time.time()
        t_all = t_all+t2-t1
    del_files(os.path.join(os.getcwd(),'temp'))

    print('正在处理'+str(frame_num)+'/'+str(frame_num)+'帧,完成了'+str((frame_num-1)/frame_num*100)+'%,预计剩余时间0s')
    if add_type == '1': #x2
        for i in range(0,2):
            shutil.copyfile(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames',str(frame_num).zfill(9)+'.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter).zfill(9)+'.png'))
            output_frame_counter = output_frame_counter+1
    elif add_type == '2': #x4
        for i in range(0,4):
            shutil.copyfile(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'original_frames',str(frame_num).zfill(9)+'.png'),os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames',str(output_frame_counter).zfill(9)+'.png'))
            output_frame_counter = output_frame_counter+1

    print('处理完成\n')
    print('开始合成视频...')
    os.makedirs(os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'output_videos'))
    
    os.system('ffmpeg -f image2 -r '+str(target_fps)+' -i '+os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'interpolated_frames','%09d.png')+' -vcodec h264 '+os.path.join(output_path,'.'.join(os.path.basename(f).split('.')[:-1]),'output_videos',str(target_fps)+'fps_'+'.'.join(os.path.basename(f).split('.')[:-1])+'.mp4'))

    print('视频合成完毕')
    
    os.system('pause')
