import torch
import time
import math
import numpy as np
from torch.backends import cudnn
from torch import optim
import os
import warnings
from torch import nn

warnings.filterwarnings("ignore")


class options:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model = True
        self.save_model = True
        self.steps = 84000
        self.num_workers = 1
        self.lambda_identity = 5
        self.lambda_cycle = 10
        self.checkpoint_model = "trained_models/checkpoint.pk"
        self.eval_step = 50
        self.lr = 0.0001
        self.gps = 3
        self.blocks = 6
        self.bs = 1
        self.crop = True
        self.crop_size = 256


opt = options()

if not os.path.exists("trained_models"):
    os.mkdir("trained_models")
if not os.path.exists("saved_images"):
    os.mkdir("saved_images")
if not os.path.exists("test_images"):
    os.mkdir("test_images")
if not os.path.exists("test_images/best_picks"):
    os.mkdir("test_images/best_picks")
if not os.path.exists("runs"):
    os.makedirs("runs", exist_ok=True)
if not os.path.exists("inputs"):
    os.makedirs("inputs", exist_ok=True)
if not os.path.exists("outputs"):
    os.makedirs("outputs", exist_ok=True)
