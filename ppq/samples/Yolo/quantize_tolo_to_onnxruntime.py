# PPQ导出模型到onnxruntime中进行推理
# 这里例子针对pytorch中的torchscript格式
import sys
sys.path.insert(0, '/home/chaizehua/ppq')

import os
import torchvision.transforms as transforms
from PIL import Image
from ppq import *
from ppq.api import *
import torch
import sys
print(sys.path)
TORCH_SCRIPT_PATH        = 'models/resnet50-v2-7.onnx'      # 你的模型位置
OUTPUT_PATH      = 'Output'                    # 生成的量化模型的位置
CALIBRATION_PATH = 'imgs'                      # 校准数据集
BATCHSIZE        = 1                           # 32
EXECUTING_DEVICE = 'cuda'
QUANT_TYPE = "resnet50-v2-7"                            # yolov5l6-INT8 FP16 or FP32
# create dataloader
imgs = []
trans = transforms.Compose([
    transforms.Resize([640, 640]),  # [h,w]
    transforms.ToTensor(),
])
os.chdir(sys.path[1])
for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img) # img is 0 - 1

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE, drop_last=True)

# m = torch.jit.load(TORCH_SCRIPT_PATH, _extra_files={'config.txt': ''}, map_location=EXECUTING_DEVICE)
# m.float()

with ENABLE_CUDA_KERNEL():
    qir = quantize_onnx_model(
        platform=TargetPlatform.TRT_INT8,
        onnx_import_file=TORCH_SCRIPT_PATH, 
        calib_dataloader=dataloader, 
        calib_steps=32, device=EXECUTING_DEVICE,
        input_shape=[BATCHSIZE, 3, 640, 640], 
        collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    # qir = quantize_torch_model(
    #     platform=TargetPlatform.TRT_INT8,
    #     model=m, 
    #     calib_dataloader=dataloader, 
    #     calib_steps=32, device=EXECUTING_DEVICE,
    #     input_shape=[BATCHSIZE, 3, 640, 640], 
    #     collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    # 网络误差分析
    snr_report = graphwise_error_analyse(
        graph=qir, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    # 层间误差分析
    snr_report = layerwise_error_analyse(
        graph=qir, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    
    export_ppq_graph(
        qir, platform=TargetPlatform.TRT_INT8, 
        graph_save_to=OUTPUT_PATH + '/' + QUANT_TYPE +  '.onnx',
        config_save_to=OUTPUT_PATH + '/' + QUANT_TYPE +  '.json')
    
    # # 生成tensorrt engine相关文件
    # from ppq.utils.TensorRTUtil import build_engine, Benchmark, Profiling
    # # build_engine(
    # #     onnx_file=OUTPUT_PATH + '/' + QUANT_TYPE +  '.onnx', 
    # #     engine_file=OUTPUT_PATH + '/' + QUANT_TYPE +  '.engine')
    # build_engine(
    #     onnx_file=OUTPUT_PATH + '/' + QUANT_TYPE +  '.onnx', 
    #     engine_file=OUTPUT_PATH + '/' + QUANT_TYPE +  '.engine', int8=True, 
    #     int8_scale_file=OUTPUT_PATH + '/' + QUANT_TYPE +  '.json')

    # Benchmark(OUTPUT_PATH + '/' + QUANT_TYPE +  '.engine')
    # Profiling(OUTPUT_PATH + '/' + QUANT_TYPE +  '.engine')