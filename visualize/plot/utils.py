"""
Copyright (c) 2023, Auorui.
All rights reserved.
"""
import cv2
import numpy as np

def OverlayPng(imgBack, imgFront, pos=(0, 0), alpha_gain=1.0):
    """
    Overlay display image with proper alpha blending
    :param imgBack: Background image, no format requirement, 3 channels
    :param imgFront: PNG pre image, must be read using cv2.IMREAD_UNCHANGED=-1
    :param pos: Placement position

    Examples:
    '''
        background = cv2.imread(background_path)
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        fused_image = pyzjr.OverlayPng(background, overlay, alpha_gain=1.5)
    '''
    """
    img_back = imgBack.copy()
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    y_pos, x_pos = pos
    y_end = y_pos + hf
    x_end = x_pos + wf

    # Ensure we don't go beyond the background boundaries
    if y_end > hb:
        y_end = hb
    if x_end > wb:
        x_end = wb

    # Resize overlay to fit the background (optional but good practice)
    overlay_resized = cv2.resize(imgFront, (x_end - x_pos, y_end - y_pos))

    overlay_alpha = overlay_resized[:, :, 3].astype(float) / 255.0
    overlay_alpha = np.clip(overlay_alpha * alpha_gain, 0, 1)
    background_alpha = 1.0 - overlay_alpha

    result = overlay_resized[:, :, :3] * overlay_alpha[..., np.newaxis] + img_back[y_pos:y_end, x_pos:x_end, :3] * background_alpha[..., np.newaxis]
    img_back[y_pos:y_end, x_pos:x_end, :3] = result.astype(np.uint8)

    return img_back
