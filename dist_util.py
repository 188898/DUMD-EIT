
import torch as th
import torch.distributed as dist


def setup_dist():

    if th.cuda.is_available(): 
        th.cuda.set_device(dev())
        th.cuda.empty_cache()

def dev():

    if th.cuda.is_available():
        return th.device(f"cuda:0")
    return th.device("cpu")
  