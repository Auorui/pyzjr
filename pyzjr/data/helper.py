import cv2
import numpy as np
from itertools import repeat
from typing import Iterable

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def norm_to_abs(cx_norm, cy_norm, w_norm, h_norm, img_w, img_h):
    """
    Convert normalized bounding box coordinates to absolute pixel coordinates.

    This function transforms bounding box parameters from normalized [0,1] range
    (common in machine learning datasets) to absolute pixel values based on image dimensions.

    :param cx_norm: Normalized x-coordinate of bounding box center [0,1]
    :param cy_norm: Normalized y-coordinate of bounding box center [0,1]
    :param w_norm: Normalized width of bounding box [0,1]
    :param h_norm: Normalized height of bounding box [0,1]
    :param img_w: Width of the image in pixels
    :param img_h: Height of the image in pixels
    :return: list: [x_min, y_min, x_max, y_max] absolute coordinates (pixels) representing:
                                                x_min: Left boundary of bounding box
                                                y_min: Top boundary of bounding box
                                                x_max: Right boundary of bounding box
                                                y_max: Bottom boundary of bounding box
    """
    cx = cx_norm * img_w
    cy = cy_norm * img_h
    w = w_norm * img_w
    h = h_norm * img_h
    x_min = cx - w/2
    y_min = cy - h/2
    x_max = cx + w/2
    y_max = cy + h/2
    return [x_min, y_min, x_max, y_max]


def abs_to_norm(x_min, y_min, x_max, y_max, img_w, img_h):
    """
    Convert absolute pixel coordinates to normalized bounding box coordinates.

    This function transforms bounding box parameters from absolute pixel values
    to normalized [0,1] range (common in machine learning datasets).

    :param x_min: Left boundary of bounding box in pixels
    :param y_min: Top boundary of bounding box in pixels
    :param x_max: Right boundary of bounding box in pixels
    :param y_max: Bottom boundary of bounding box in pixels
    :param img_w: Width of the image in pixels
    :param img_h: Height of the image in pixels
    :return: list: [cx_norm, cy_norm, w_norm, h_norm] normalized coordinates representing:
                    cx_norm: Normalized x-coordinate of bounding box center [0,1]
                    cy_norm: Normalized y-coordinate of bounding box center [0,1]
                    w_norm: Normalized width of bounding box [0,1]
                    h_norm: Normalized height of bounding box [0,1]
    """
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [cx_norm, cy_norm, w_norm, h_norm]


def _default_palette(num_classes):
    """Generate a fixed palette for num_classes 0: (r,g,b), ..."""
    palette = {}
    for cid in range(num_classes):
        if num_classes == 2:
            palette[cid] = (0, 0, 0) if cid == 0 else (255, 255, 255)
        else:
            palette[cid] = ((cid * 97) % 256, (cid * 57) % 256, (cid * 23) % 256)
    return palette


def index2color(index_map, num_classes, color_map, rgb_output=True):
    """Map an (H,W) integer index map to an (H,W,3) uint8 color image.

    Args:
        index_map: np.ndarray, (H,W) integer class indices in [0, num_classes).
        color_map: dict {class_id(int or str): (r,g,b)}.
        num_classes: number of classes.
        rgb_output: if True return RGB; if False return BGR (ready for cv2.imwrite).

    Returns:
        np.ndarray (H,W,3) uint8 color image.
    """
    index_map = np.asarray(index_map)
    if index_map.ndim != 2:
        raise ValueError(f"index_map must be (H,W), got {index_map.shape}")

    # build palette: (num_classes, 3) RGB
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k, v in color_map.items():
        cid = int(k)
        if not 0 <= cid < num_classes:
            raise ValueError(f"Invalid class id: {cid}")
        color = np.asarray(v, dtype=np.uint8)
        if color.shape != (3,):
            raise ValueError(f"Color for class {cid} must be RGB triplet, got {color.shape}")
        palette[cid] = color

    # validate index range
    if index_map.min() < 0 or index_map.max() >= num_classes:
        bad = np.unique(index_map[(index_map < 0) | (index_map >= num_classes)])[:10]
        raise ValueError(f"index_map contains values outside [0,{num_classes}): {bad.tolist()}")

    color = palette[index_map]              # (H,W,3) RGB
    if not rgb_output:
        color = color[:, :, ::-1].copy()    # RGB -> BGR
    return color


def logits2color(logits, num_classes, color_map, rgb_output=True):
    """Convenience: (B,C,H,W) logits -> list of (H,W,3) color images.

    Args:
        logits: torch.Tensor or np.ndarray, (B,C,H,W) or (C,H,W) or (H,W).
        color_map: dict {class_id: (r,g,b)}.
        num_classes: number of classes.
        rgb_output: if True return RGB; if False return BGR.

    Returns:
        If input is (B,C,H,W): list of B color images.
        If input is (C,H,W): single color image.
    """
    # torch -> numpy, no grad
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits)

    color_map = _default_palette(num_classes) if color_map is None else color_map

    if logits.ndim == 4:                       # (B,C,H,W)
        preds = logits.argmax(axis=1)          # (B,H,W)
        return [index2color(p, num_classes, color_map, rgb_output) for p in preds]
    elif logits.ndim == 3:                     # (C,H,W)
        pred = logits.argmax(axis=0)           # (H,W)
        return index2color(pred, num_classes, color_map, rgb_output)
    elif logits.ndim == 2:                     # already (H,W)
        return index2color(logits, num_classes, color_map, rgb_output)
    else:
        raise ValueError(f"logits must be (B,C,H,W)/(C,H,W)/(H,W), got {logits.shape}")


def error2color(pred_mask, gt_mask, ignore_index=None, rgb_output=True, palette=None):
    """Map prediction-vs-GT error states to a color image (binary / one-vs-rest).

    Args:
        pred_mask: (H,W) integer prediction indices.
        gt_mask:   (H,W) integer ground-truth indices.
        ignore_index: value in gt_mask to ignore (e.g. 255/256); None = no ignore.
        rgb_output: True -> RGB, False -> BGR (for cv2.imwrite).
        palette: optional (5,3) uint8 array in RGB. If None, uses:
            [0] TN  -> (0,0,0)       black
            [1] TP  -> (255,255,255) white
            [2] FP  -> (255,0,0)     red
            [3] FN  -> (0,255,0)     green
            [4] ignore -> (128,128,128) gray

    Returns:
        (H,W,3) uint8 color image.
    """
    pred_mask = np.asarray(pred_mask)
    gt_mask = np.asarray(gt_mask)
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

    if not np.all(np.isin(np.unique(pred_mask), [0, 1])):
        raise ValueError(f"pred_mask must be binary (0/1), "
                         f"got {np.unique(pred_mask).tolist()}")
    gt_valid = gt_mask[gt_mask != ignore_index] if ignore_index is not None else gt_mask
    if not np.all(np.isin(gt_valid, [0, 1])):
        raise ValueError(f"gt_mask must be binary (0/1) except ignore_index={ignore_index}, "
                         f"got {np.unique(gt_mask).tolist()}")

    # resolve palette
    if palette is None:
        palette = np.array([
            [0, 0, 0],       # TN black
            [255, 255, 255], # TP white
            [255, 0, 0],     # FP red
            [0, 255, 0],     # FN green
            [128, 128, 128], # ignore gray
        ], dtype=np.uint8)
    else:
        palette = np.asarray(palette, dtype=np.uint8)
        if palette.shape != (5, 3):
            raise ValueError(f"palette must be (5,3) RGB, got {palette.shape}")

    # states: 0=TN, 1=TP, 2=FP, 3=FN, 4=ignore
    state = np.zeros_like(gt_mask, dtype=np.uint8)
    state[(pred_mask == 1) & (gt_mask == 1)] = 1   # TP
    state[(pred_mask == 1) & (gt_mask == 0)] = 2   # FP
    state[(pred_mask == 0) & (gt_mask == 1)] = 3   # FN
    if ignore_index is not None:
        state[gt_mask == ignore_index] = 4         # ignore overrides

    color = palette[state]              # (H,W,3) RGB
    if not rgb_output:
        color = color[:, :, ::-1].copy()  # RGB -> BGR
    return color


def label2index(label, num_classes, color_map, ignore_index=256, label_threshold=127, strict=True):
    """Convert various label formats to an (H,W) int64 index map.

    Accepted inputs:
        - (H,W) valid index map            -> returned as-is
        - (H,W) {0,1} binary map           -> returned as-is
        - (H,W) grayscale map              -> thresholded to {0,1}
        - (H,W,3) color label map          -> mapped via color_map
          (binary case with >2 unique colors is thresholded first)

    Args:
        label: np.ndarray, (H,W) or (H,W,3).
        num_classes: number of classes.
        color_map: dict {class_id(int): (r,g,b)}.
        ignore_index: value for ignored pixels.
        label_threshold: threshold for grayscale/binary conversion.
        strict: if True, raise on colors not present in color_map.

    Returns:
        (H,W) int64 index map.
    """
    label = np.asarray(label)
    if color_map is None:
        if label.ndim == 3:
            if label.shape[-1] != 1:
                raise ValueError(
                    f"color_map is None but got 3-channel label {label.shape}; "
                    f"pass color_map for color labels"
                )
            label = label[..., 0]
        if label.ndim != 2:
            raise ValueError(f"Unsupported label shape: {label.shape}")
        # Retain ignore_index, and the rest must fall within [0, num_classes)
        valid = (label == ignore_index) | ((label >= 0) & (label < num_classes))
        if strict and not np.all(valid):
            bad = np.unique(label[~valid])[:10]
            raise ValueError(f"Label values out of range [0,{num_classes}): {bad.tolist()}")
        return label.astype(np.int64)
    # normalize color_map: int keys, uint8 (3,) values
    cmap = {}
    for k, v in color_map.items():
        cid = int(k)
        color = np.asarray(v, dtype=np.uint8)
        if color.shape != (3,):
            raise ValueError(f"Color for class {cid} must be RGB triplet")
        cmap[cid] = color

    # 1) 2D input
    if label.ndim == 2:
        if np.all((label == ignore_index) |
                  ((label >= 0) & (label < num_classes))):
            return label.astype(np.int64)
        if label.max() <= 1:
            return label.astype(np.int64)
        return (label >= label_threshold).astype(np.int64)

    if label.ndim != 3 or label.shape[-1] != 3:
        raise ValueError(f"Unsupported label shape: {label.shape}")

    # 2) binary color label with >2 unique colors -> threshold first
    if num_classes == 2:
        unique_colors = np.unique(label)
        if len(unique_colors) > 2:
            gray = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            binary = (gray >= label_threshold).astype(np.uint8) * 255
            label = np.repeat(binary[..., None], 3, axis=-1)

    # 3) color lookup
    result = np.full(label.shape[:2], ignore_index, dtype=np.int64)
    for cid, color in cmap.items():
        result[np.all(label == color, axis=-1)] = cid

    # 4) strict check
    if strict:
        unknown = result == ignore_index
        if np.any(unknown):
            unknown_colors = np.unique(label[unknown].reshape(-1, 3), axis=0)[:10]
            raise ValueError(f"Unknown label colors: {unknown_colors.tolist()}")

    return result