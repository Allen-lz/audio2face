import torch
import torch.nn as nn
import onnx
import numpy as np
from models.mobilenetv3 import *
from configs import cfg


model = mobilenetv3_small(num_classes=cfg.num_classes, width_mult=cfg.width_mult, output_c=cfg.output_c).to(
            'cuda:0')


ckpt = torch.load(r'E:\workplace\ARKit\checkpoints\iteration_8000.pt')
model.load_state_dict(ckpt['state_dict'], strict=True)
model.eval()

input_names = ["input0"]
output_names = ["output0", "output1", "output2", "output3"]

x = torch.tensor(np.zeros((1, 3, 224, 224)).astype(np.float32)).cuda()

torch.onnx.export(model, x, 'model.onnx', input_names=input_names, output_names=output_names,)
# 指定input_0和output_0的batch可变
