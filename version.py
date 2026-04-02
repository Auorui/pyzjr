import os
import torch
import sys
import platform

__version__ = '1.4.21'

def pyzjr_info():
    print("Python Version:", ".".join(map(str, [sys.version_info.major,
                                          sys.version_info.minor,
                                          sys.version_info.micro])))
    print("Pytorch Version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("CPU Model:", platform.processor())
    print("Number Of CPU Cores:", os.cpu_count())
    print("System Information:", platform.system() + " " + platform.version())
    print("-------------------------------------")
    print("Pyzjr Version:", __version__)

if __name__=="__main__":
    pyzjr_info()