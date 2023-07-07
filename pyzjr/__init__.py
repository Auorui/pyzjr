"""
- author:Auorui(夏天是冰红茶)
- creation time:2022.10
- pyzjr is the Codebase accumulated by my python programming.
  At present, it is only for my personal use. If you want to
  use it, please contact me. Here are my email and WeChat.
- WeChat: z15583909992
- Email: zjricetea@gmail.com
- Note: Currently still being updated, please refer to the latest version for any changes that may occur
"""

"""
from pyzjr import pyps as ps  # Using Image Processing
from pyzjr import pysift as sift  # Using SIFT matching algorithm
from pyzjr import Color as color
"""

__version__ = "0.0.9"

import cv2
from pyzjr.PIC import download_file,getPhotopath,Pic_rename,read_resize_image,\
                      load_images_from_folder,save_images
from pyzjr.Enimage import Filter,Enhance,Random_Enhance,Retinex,\
                          repair_Img
from pyzjr.definition import Fuzzy_image,Clear_quantification
from pyzjr.ColorModule import *
from pyzjr.definition import *
from pyzjr.TrackBar import *
from pyzjr.utils import *
from pyzjr.video import *
from pyzjr.Z import * 

repair_TELEA=cv2.INPAINT_TELEA
repair_NS=cv2.INPAINT_NS
Lap_64F=cv2.CV_64F

print("test successful")









