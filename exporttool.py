import time

import onnxruntime
import torch
from model_hub import Alexnet_modify, mobilenet_v2_modify, Resnet18_modify, Shufflenet_v205_modify, squeezenet_modify, \
    Google_net_modify

# net = Alexnet_modify.Alex_net(num_classes=3)
# # net = Google_net_modify.googlenet(num_classes=3)
# export_onnx_file = './onxs/alexnet.onnx'
x = torch.randn(1, 3, 224, 224)  # 生成张量
# torch.onnx.export(net,
#                   x,
#                   export_onnx_file,
#                   opset_version=11,
#                   verbose=True,
#                   do_constant_folding=True,  # 是否执行常量折叠优化
#                   input_names=["input"],  # 输入名
#                   output_names=["output"])
# import onnxsim
#
# model_onnx, check = onnxsim.simplify('./onxs/alexnet.onnx')
#
import netron

modelData = "./onxs/alexnet.onnx"
netron.start(modelData)

#
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#
# # 模型加载
# # 方式1：
#
# # code


# onnx_model_path = "./onxs/best.onnx"
# resnet_session = onnxruntime.InferenceSession(onnx_model_path)
# inputs = {resnet_session.get_inputs()[0].name: to_numpy(x)}
# time_start_1 = time.time()
# outs = resnet_session.run(None, inputs)[0]
# print("onnx prediction", outs.argmax(axis=1)[0])
# time_end_1 = time.time()
# print("运行时间："+str(time_end_1 - time_start_1)+"秒")
