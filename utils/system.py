import os
import numpy as np
import random
import torch

def reproduce(seed):
    # 设置 Python 哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python 内置随机
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDNN 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
