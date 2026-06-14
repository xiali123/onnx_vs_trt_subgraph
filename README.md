# ONNX Probe

Probe and dump intermediate tensor values from ONNX models.
TRTEXEC_CMD="trtexec \
    --onnx=$ONNX_PATH \
    --saveEngine=$ENGINE_PATH \
    --timingCacheFile=$TIMING_CACHE_PATH \
    --stronglyTyped \
    --builderOptimizationLevel=3 \
    --maxAuxStreams=0 \
    --exportProfile=$PROFILE_PATH \
    --exportLayerInfo=$LAYER_INFO_PATH \
    --profilingVerbosity=detailed \
    --dumpProfile \
    --staticPlugins=/home/kevin.xia/libNv12ToRgbPlugin.so \
    --separateProfileRun \
    --verbose"









1，对整网onnx分批ort推理输出真值，

2， onnx 转trt 并存储相关层信息
3， 通过层信息，调用src\onnx_trt_map 模块生成映射关系
for :
    1. 子图划分 src\subgraph_split_by_trt 用这个模块的功能，然后选择一个数据流最上层的子图进行onnx 
    if onnx 节点数量 >= 20 个:
        1. 将选择的子图onnx 推理和用真值推理（数据处理模块处理真值符合trt推理）
        2. 在用调用src\onnx_trt_map 模块生成子图trt onnx映射关系
        3. 调用src\compare_onnx_trt 分析差异，如果值大于阈值，需要再对这个onnx进行拆分,回到 for循环开头
    else:
        4. 构建并推理onnx 带--markUnfusedTensorsAsDebugTensors 和用真值推理（数据处理模块处理真值符合trt推理）
        5. 在用调用src\onnx_trt_map 模块生成子图trt onnx映射关系
        6. 调用src\compare_onnx_trt 分析差异，如果值大于阈值，记录报告



 前置步骤（仅一次）:
   S1. 全量 ONNX → ORT 分批推理 → 全量 ground truth npy + name_mapping.json
   S2. 全量 ONNX → TRT 引擎 + 全量 layer info JSON
   S3. 全量 LayerMapper (作为整体参考)

 递归验证函数 verify(onxx_path, layer_info_path, ground_truth_dir, depth):
   for:  (对当前 onxx_path + layer_info 进行子图划分验证)
     A. TRTPartitioner(onxx_path, layer_info_path).split(sub_dir)
        产出: subgraph_0.onnx, subgraph_1.onnx, ... + partition.csv

     B. 选择最上游子图 (subgraph_0.onnx)，统计其 ONNX 节点数

     IF 节点数 >= 20:
       A1. 将子图 ONNX → TRT 引擎 + 层信息 (生成新的 layer_info JSON)
           TRTBuilder(subgraph_0.onnx,
               export_layer_info=sub_layer_info,
               save_debug_tensors=True).build(load_inputs=bin输入)
           产出: subgraph engine + sub_layer_info.json + TRT debug dump npy
       A3. data_op.NpyToBin(ground_truth_dir) 转换子图输入 → .bin

       A4. LayerMapper(subgraph_0.onnx, sub_layer_info.json)
           产出: 子图 TRT↔ONNX 映射

       A5. LayerComparator(subgraph_0.onnx, sub_layer_info, ort_dump_dir, trt_dump_dir)
           .compare() → 分析差异

       A6. 若 max_abs_diff > threshold:
             递归 → verify(subgraph_0.onnx, sub_layer_info.json, ground_truth_dir, depth+1)
             (这里 subgraph_0.onnx + sub_layer_info.json 作为下一个 for 的拆分依据)

     ELSE (节点数 < 20):
       B1. data_op.NpyToBin(ground_truth_dir) 转换子图输入 → .bin

       B2. TRTBuilder(subgraph_0.onnx,
               mark_debug_tensors=True,
               save_debug_tensors=True).build(load_inputs=bin输入)
           产出: TRT debug dump npy

       B3. LayerMapper(subgraph_0.onnx, sub_layer_info.json)

       B4. LayerComparator 对比

       B5. 若超阈值 → 记录详细报告到 failed 列表 (不再拆分)
       若通过 → 记录到 passed 列表




C:\TensorRT-10.13.0.35\bin\trtexec.EXE  --loadEngine=s.engine --saveAllDebugTensors=numpy --iterations=1 --loadInputs="input:verify_out_v1/depth_0/verify_sub_0/inputs/input.bin"




C:\TensorRT-10.13.0.35\bin\trtexec.EXE --onnx=C:\Users\Administrator\Desktop\python脚本\examples\verify_out_v1\depth_0\split\subgraph_0.onnx --saveEngine=s.engine --markUnfusedTensorsAsDebugTensors --saveAllDebugTensors=numpy --fp16  --iterations=1 --loadInputs="input:verify_out_v1/depth_0/verify_sub_0/inputs/input.bin"