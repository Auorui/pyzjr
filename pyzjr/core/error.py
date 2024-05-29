from pyzjr.core.general import is_pil, is_rgb_image, is_numpy, is_tensor, is_gray_image

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

def _check_img_is_opencv(img):
    if not is_numpy(img):
        raise ValueError("The loaded image should conform to the format read by OpenCV!")

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

def assert_shape(a, b):
    assert a.shape == b.shape, "Shape mismatch: {} and {}".format(
        a.shape, b.shape)

def assert_size(a, b):
    assert a.size == b.size, "Size mismatch: {} and {}".format(
        a.size, b.size)
