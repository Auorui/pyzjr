from .general import is_pil, is_gray_image, is_rgb_image, is_numpy, is_tensor, is_gray_image

__all__=["not_rgb_warning","Error_flags","Augment_Error_Message", "_check_img_is_plt", "_check_parameter_is_tuple_2", "_check_parameter_is_tuple_and_list_2", \
         "_check_parameter_is_tuple_and_list_or_single_2", "check_dtype", "_check_img_is_ndarray", "_check_input_is_tensor" ,"get_image_size",
         "_get_image_num_channels"]
Error_flags = "[pyzjr error]:"

Augment_Error_Message = "Image should be PIL image. Got {}. Use Decode() for encoded data or ToPIL() for decoded data."
Ndarray_Error_Message = "The input image should be of type numpy.ndarray or PIL.Image.Image. Got {}."

def not_rgb_warning(image):
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_gray_image(image):
            message += "\nYou can convert your gray image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        raise ValueError(message)

def _check_img_is_plt(img):
    if not is_pil(img):
        raise TypeError(Augment_Error_Message.format(type(img)))

def _check_img_is_ndarray(img):
    if not is_numpy(img):
        raise TypeError(Ndarray_Error_Message.format(type(img)))

def _check_input_is_tensor(img):
    if not is_tensor(img):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(img)))

def _check_parameter_is_tuple_2(a):
    if not isinstance(a, tuple) or len(a) != 2:
        raise ValueError(f"{a} should be a tuple (x_para,y_para).")

def _check_parameter_is_tuple_and_list_2(a):
    if not isinstance(a, (list, tuple)) or len(a) != 2:
        raise ValueError(f"{a} should be a list or tuple of two values.")

def _check_parameter_is_tuple_and_list_or_single_2(a):
    if not (isinstance(a, int) or (isinstance(a, (list, tuple)) and len(a) == 2)):
        raise TypeError('Should be a single number or a list/tuple (h, w) of length 2.'
                        'Got {}.'.format(a))

def get_image_size(image):
    if is_numpy(image):
        h, w = image.shape[:2]
        return h, w
    if is_pil(image):
        w, h = image.size
        return h, w
    if is_tensor(image):
        if len(image.shape) == 4 or len(image.shape) == 3:
            w, h = image.shape[-2:]
            return h, w
    else:
        raise ValueError("[pyzjr]:Unsupported input type")

def _get_image_num_channels(img):
    if is_tensor(img):
        if img.ndim == 2:
            return 1
        elif img.ndim > 2:
            return img.shape[-3]
    if is_pil(img):
        return 1 if img.mode == 'L' else 3
    if is_numpy(img):
        return 1 if is_gray_image else 3


class check_dtype():
    def __init__(self, para):
        self.para = para
    def is_int(self):
        return isinstance(self.para, int)
    def is_float(self):
        return isinstance(self.para, float)
    def is_list(self):
        return isinstance(self.para, list)
    def is_tuple(self):
        return isinstance(self.para, tuple)
    def is_list_or_tuple(self):
        return isinstance(self.para, (list, tuple))