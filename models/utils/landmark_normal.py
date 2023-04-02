import numpy as np
def norm_input_face(shape_3d, phrase="train"):
    """
    Args:
        shape_3d: 没有进行归一化
        face_std: 是进行了归一化之后的
    Returns:

    """
    shape_3d = shape_3d.reshape(-1, 68, 3)
    scale = 1.6 / (shape_3d[:, 0, 0] - shape_3d[:, 16, 0])  # 计算脸宽
    shift = - 0.5 * (shape_3d[:, 0, 0:2] + shape_3d[:, 16, 0:2])  # 计算出脸的中心位置
    depth_abs = np.max(shape_3d[:, :, -1], axis=-1) - np.min(shape_3d[:, :, -1], axis=-1)

    scale = scale[:, np.newaxis, np.newaxis]
    shift = shift[:, np.newaxis, :]
    depth_abs = depth_abs[:, np.newaxis]

    # 这里是标准化, 只是因为scale使用的是倒数, shift使用减数, 所以会写成这样
    shape_3d[:, :, 0:2] = (shape_3d[:, :, 0:2] + shift) * scale

    shape_3d[:, :, -1] = shape_3d[:, :, -1] / depth_abs

    if phrase == "train":
        return shape_3d.reshape(-1, 204)
    else:
        return shape_3d.reshape(-1, 204), scale, shift