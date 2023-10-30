import numpy as np

__all__ = ["getPalette", "colorstr"]

VOC_COLOR = [
             [0, 0, 0],    [128, 0, 0],   [0, 128, 0],    [128, 128, 0],
             [0, 0, 128],  [128, 0, 128], [0, 128, 128],  [128, 128, 128],
             [64, 0, 0],   [192, 0, 0],   [64, 128, 0],   [192, 128, 0],
             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
             [0, 64, 0],   [128, 64, 0],  [0, 192, 0],    [128, 192, 0],
             [0, 64, 128]
             ]

def getPalette(color=VOC_COLOR):
    pal = np.array(color, dtype='uint8').flatten()
    return pal

def colorstr(*input):
    # https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']