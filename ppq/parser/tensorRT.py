# This file is created by Nvidia Corp.
# Modified by PPQ develop team.
#
# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import struct
from typing import List

from ppq.core import (NetworkFramework, QuantizationPolicy,
                      QuantizationProperty, ppq_info, ppq_warning)
from ppq.IR import BaseGraph, GraphExporter
from ppq.IR.quantize import QuantableOperation

# from .caffe_exporter import CaffeExporter
from .onnxruntime_exporter import OnnxExporter, ONNXRUNTIMExporter


class TensorRTExporter_QDQ(ONNXRUNTIMExporter):
    """
    TensorRT PPQ 0.6.4 以来新加入的功能
    
    你需要注意，只有 TensorRT 8.0 以上的版本支持读取 PPQ 导出的量化模型
    并且 TensorRT 对于量化模型的解析存在一些 Bug，
    
    如果你遇到模型解析不对的问题，欢迎随时联系我们进行解决。
    
        已知的问题包括：
        1. 模型导出时最好不要包含其他 opset，如果模型上面带了别的opset，比如 mmdeploy，trt有可能会解析失败
        2. 模型导出时可能出现 Internal Error 10, Invalid Node xxx()，我们还不知道如何解决该问题
    
    Args:
        ONNXRUNTIMExporter (_type_): _description_
    """

    def export(self, file_path: str, graph: BaseGraph,
               save_as_external_data: bool = False) -> None:
        # step 1, export onnx file.
        super().export(
            file_path=file_path, graph=graph, quantized_param=False,
            config_path=None, save_as_external_data=save_as_external_data)


class TensorRTExporter_JSON(GraphExporter):
    """
    使用该导出器导出符合 TensorRT 格式要求的量化模型。
    
    PPQ 可以将 TQC 中的量化信息导出到 json，并遵循 TensorRT 支持的格式。
    用户可以通过 Json 文件将图中所有 variable 的 min, max 传递给 TensorRT           # 用户可以通过 Json 文件将图中所有 variable 的 min, max 传递给 TensorRT 所有未写入 min, max 的变量将保持 fp32(fp16) 精度执行
    所有未写入 min, max 的变量将保持 fp32(fp16) 精度执行。
    """
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        act_quant_info = {}
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation): continue
            if op.type in {"Gather", "Unsqueeze", "Concat", "Reshape", "Squeeze"}: continue     # 这些算子不进行量化

            for cfg, var in op.config_with_variable:
                if not cfg.can_export(export_overlapped=True): continue
                if var.is_parameter: continue       # parameter的不导出

                if cfg.policy != QuantizationPolicy(            # TensorRT只导出量化config是QuantizationProperty.LINEAR + QuantizationProperty.SYMMETRICAL + QuantizationProperty.PER_TENSOR的算子
                    QuantizationProperty.LINEAR + 
                    QuantizationProperty.SYMMETRICAL + 
                    QuantizationProperty.PER_TENSOR):
                    ppq_warning(f'Can not export quantization config on variable {var.name}, '
                                'Quantization Policy is invalid.')
                    continue

                if cfg.num_of_bits != 8 or cfg.quant_max != 127 or cfg.quant_min != -128:           # 目前只支持int8量化 -128，127区间内
                    ppq_warning(f'Can not export quantization config on variable {var.name}, '
                                'Tensor Quantization Config has unexpected setting.')
                    continue

                act_quant_info[var.name] = cfg.scale.item() * 127                                   # act_quant_info 的value值为 scale * 127， 即区间实际的scale的大小 从这个[-quant_scale, quant_scale] 映射到[-128， 127]

        json_qparams_str = json.dumps({'act_quant_info': act_quant_info}, indent=4)
        with open(config_path, "w") as json_file:
            json_file.write(json_qparams_str)

    def export_weights(self, graph: BaseGraph, config_path: str = None):
        topo_order =  graph.topological_sort()
        weights_list = []
        for index, op in enumerate(topo_order):
            if op.type in {"Conv", "Gemm"}:
                weights_list.extend(op.parameters)

        weight_file_path = os.path.join(os.path.dirname(config_path), "quantized.wts")              # 全部的value值打印出来送进去（未量化）

        f = open(weight_file_path, 'w')
        f.write("{}\n".format(len(weights_list)))

        for param in weights_list:
            weight_name = param.name
            weight_value = param.value.reshape(-1).cpu().numpy()
            f.write("{} {}".format(weight_name, len(weight_value)))
            for value in weight_value:
                f.write(" ")
                f.write(struct.pack(">f", float(value)).hex())
            f.write("\n")
        ppq_info(f'Parameters have been saved to file: {weight_file_path}')


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        ppq_info(
            'You are exporting PPQ Graph to TensorRT(Onnx + Json). \n'
            'Please Compile the TensorRT INT8 engine manually: \n\n'
            'from ppq.utils.TensorRTUtil import build_engine \n'
            "build_engine(onnx_file='Quantized.onnx', int8_scale_file='Quantized.json', engine_file='Quantized.engine', int8=True)\n")
        if config_path is not None:
            self.export_quantization_config(config_path, graph)
            self.export_weights(graph, config_path)
        _, ext = os.path.splitext(file_path)
        if ext == '.onnx':
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)
        elif ext in {'.prototxt', '.caffemodel'}:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)

        # no pre-determined export format, we export according to the
        # original model format
        elif graph._built_from == NetworkFramework.CAFFE:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)
        elif graph._built_from == NetworkFramework.ONNX:
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)
