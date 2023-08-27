repair_NS = 0
repair_TELEA = 1
Lap_64F = 6
# video
Esc = 27
# BGR
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)
grey = (192, 192, 192)
white = (255, 255, 255)
yellow = (0, 255, 255)
orange = (0, 97, 255)
purple = (255, 0, 255)
violet = (240, 32, 160)
# RGB
voc = [(0, 0, 0),    (128, 0, 0),    (0, 128, 0),   (128, 128, 0), 
       (0, 0, 128),  (128, 0, 128),  (0, 128, 128), (128, 128, 128),
       (64, 0, 0),   (192, 0, 0),    (64, 128, 0),  (192, 128, 0), 
       (64, 0, 128), (192, 0, 128),  (64, 128, 128), (192, 128, 128),
       (0, 64, 0),   (128, 64, 0),   (0, 192, 0),   (128, 192, 0), 
       (0, 64, 128), (128, 64, 12)]
# HSV
class HSV:
       """
       * 示例仅供参考，只是为了在调试过程中能够更快的找到相要的颜色
       可以进行调试成功的值进行替换，但也要记住将原来颜色的值注释。
       """
       def __init__(self, mode=False):
              """
              初始化HSV类，可以选择是否启用转换模式。
              :param mode: 如果为True，启用转换模式。
              """
              # hmin, smin, vmin, hmax, smax, vma
              self.COLORS = {
                     "black": [0, 0, 0, 179, 255, 46],
                     "gray": [0, 0, 46, 179, 43, 220],
                     "white": [0, 0, 221, 179, 30, 225],
                     "red": [156, 43, 46, 179, 255, 225],
                     "red2": [0, 43, 46, 10, 255, 225],
                     "orange": [11, 43, 46, 25, 255, 225],
                     "yellow": [26, 43, 46, 34, 255, 225],
                     "green": [28, 38, 36, 64, 255, 255],
                     "cyan": [78, 43, 46, 99, 255, 225],
                     "blue": [100, 43, 46, 124, 255, 225],
                     "purple": [125, 43, 46, 155, 255, 225],
              }

              self.mode = mode

       def trans(self, orlist):
              """
              将颜色列表转换为包含两个子列表的列表。
              :param orlist: 输入的颜色列表。
              :return: 两个子列表组成的列表。
              """
              if self.mode:
                     result_list = [orlist[:3], orlist[3:]]
              else:
                     result_list = orlist
              return result_list

       def __getattr__(self, color_name):
              """
              获取颜色属性，并根据模式执行转换。
              :param color_name: 颜色名称。
              :return: 转换后的颜色列表或原始颜色列表，取决于模式。
              """
              if color_name in self.COLORS:
                     return self.trans(self.COLORS[color_name])
              else:
                     raise AttributeError(f"[pyzjr]:'HSV' object has no attribute '{color_name}'")

# HandLandmark
class HandLandmark():
       WRIST = 0
       THUMB_CMC = 1
       THUMB_MCP = 2
       THUMB_IP = 3
       THUMB_TIP = 4                   # TIP
       INDEX_FINGER_MCP = 5
       INDEX_FINGER_PIP = 6
       INDEX_FINGER_DIP = 7
       INDEX_FINGER_TIP = 8             # TIP
       MIDDLE_FINGER_MCP = 9
       MIDDLE_FINGER_PIP = 10
       MIDDLE_FINGER_DIP = 11
       MIDDLE_FINGER_TIP = 12            # TIP
       RING_FINGER_MCP = 13
       RING_FINGER_PIP = 14
       RING_FINGER_DIP = 15
       RING_FINGER_TIP = 16              # TIP
       PINKY_MCP = 17
       PINKY_PIP = 18
       PINKY_DIP = 19
       PINKY_TIP = 20                    # TIP