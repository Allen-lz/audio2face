import numpy as np
import cv2, os, sys, torch



import sys
sys.path.append(".")
from PIL import Image
from models._3dmm.face3d.util.preprocess import align_img
from models._3dmm.face3d.util.load_mats import load_lm3d
from models._3dmm.face3d.models import networks
from models._3dmm.face3d.extract_kp_videos import KeypointExtractor


from video_analysis_and_data_generation import FaceProcess

from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def get_rotation_matrix(yaw, pitch, roll):

    # 角度值转弧度制
    # yaw = yaw / 180 * 3.14
    # pitch = pitch / 180 * 3.14
    # roll = roll / 180 * 3.14

    yaw = torch.FloatTensor([yaw])
    pitch = torch.FloatTensor([pitch])
    roll = torch.FloatTensor([roll])

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    # 旋转角转旋转矩阵
    # https://zhuanlan.zhihu.com/p/82142292
    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw),
                         torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                         -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    # 旋转矩阵连乘合并为一个
    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1])
    zeros = torch.zeros([batch_size, 1])
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x

    # 这里的转置就像是求逆一样
    return rot.permute(0, 2, 1)


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]

    # print(exp_coeffs.shape)
    # print(angles.shape)
    # print(translations.shape)
    # print("--------------------------------------------------")
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }


def vis_2dpoints(img, points):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.array(img)

    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)
    plt.imshow(img)
    plt.show()


def _2d_vis(points):

    p1 = points[:, 0]  # 数据点
    p2 = points[:, 1]
    # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
    plt.figure('Draw')
    plt.scatter(p1, p2)  # scatter绘制散点图
    # plt.draw()  # 显示绘图
    plt.show()


def _3d_vis(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2], zdir='z', c='c')
    plt.show()


class FaceRecon():
    def __init__(self, path_of_net_recon_model, dir_of_BFM_fitting, device="cuda"):
        self.face_process = FaceProcess()

        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
        checkpoint = torch.load(path_of_net_recon_model, map_location=torch.device(device))
        self.net_recon.load_state_dict(checkpoint['net_recon'])
        self.net_recon.eval()
        self.lm3d_std = load_lm3d(dir_of_BFM_fitting)
        self.device = device

    def run(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm3d, face_img = self.face_process.detect_facelandmark(img_rgb)

        face_img = Image.fromarray(np.uint8(face_img))

        lm = lm3d[:, :2]

        W, H = face_img.size

        lm1 = lm.reshape([-1, 2])

        lm1[:, -1] = H - 1 - lm1[:, -1]  # 将关键点沿着水平线翻个面

        # vis_2dpoints(np.array(face_img)[::-1, :, :], lm1)  # 正确

        trans_params, im1, lm1, _ = align_img(face_img, lm1, self.lm3d_std)

        print(lm1.shape)

        # vis_2dpoints(np.array(im1)[::-1, :, :], lm1)  # 正确

        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_t = torch.tensor(np.array(im1) / 255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # 放进网络中去进行系数的预测, 而且只要得到这些系数就行了, im_t是进行系数预测的, 但是lm1其实是进行loss的计算的, 在推理的时候用不上
            full_coeff = self.net_recon(im_t)
            coeffs = split_coeff(full_coeff)

        pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
        rot_mat = compute_rotation(torch.FloatTensor(pred_coeff["angle"]))
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, torch.FloatTensor(lm3d[np.newaxis, :, :]))[0]

        kp_rotated = np.array(kp_rotated, dtype=np.int32)

        # 通过这个可视化可以发现, 3DMM预测出来的角度转换得到旋转矩阵的转置(逆)确实可以将关键点转正, 但是这个转正实在3DMM的一个隐空间转正的, 映射不回原图上
        vis_2dpoints(np.array(face_img)[::-1, :, :], kp_rotated)

        # _2d_vis(kp_rotated)

        pred_coeff = np.concatenate([
            pred_coeff['exp'],
            pred_coeff['angle'],
            pred_coeff['trans'],
            trans_params[2:][None],  # s, t
        ], 1)  # (1, 73)






if __name__ == "__main__":
    face_recon = FaceRecon("D:/audio2face/SadTalker/checkpoints/epoch_20.pth",
                           "D:/audio2face/SadTalker/checkpoints/BFM_Fitting")

    filename = "examples/face_1.png"
    image_bgr = cv2.imread(filename)

    face_recon.run(image_bgr)


