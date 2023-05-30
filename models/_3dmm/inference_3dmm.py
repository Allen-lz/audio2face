import numpy as np
import cv2, os, sys, torch



import sys
sys.path.append(".")
from PIL import Image
from models._3dmm.face3d.util.preprocess import align_img, align_img_tensor
from models._3dmm.face3d.util.load_mats import load_lm3d
from models._3dmm.face3d.models import networks
from models._3dmm.face3d.extract_kp_videos import KeypointExtractor
from models._3dmm.bfm import ParametricFaceModel

from video_analysis_and_data_generation import FaceProcess

from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
# renderer parameters
parser.add_argument('--focal', type=float, default=1015.)
parser.add_argument('--center', type=float, default=112.)  # 如果我直接更改这个东西
parser.add_argument('--camera_d', type=float, default=10.)
parser.add_argument('--z_near', type=float, default=5.)
parser.add_argument('--z_far', type=float, default=15.)  # 15
parser.add_argument('--bfm_folder', type=str, default="models/_3dmm/BFM")  # 15

opt_parser = parser.parse_args()

def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian

    这里的angle对应的应该是弧度制
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
    # return rot


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
    H, W = img.shape[:2]
    points[:, 1] = H - points[:, 1]

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

    def run_test(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm3d, face_img = self.face_process.detect_facelandmark(img_rgb)

        face_img = Image.fromarray(np.uint8(face_img))

        lm_3d = lm3d.copy()

        lm = lm_3d[:, :2]
        W, H = face_img.size
        lm1 = lm.reshape([-1, 2])
        lm1[:, -1] = H - 1 - lm1[:, -1]
        # vis_2dpoints(np.array(face_img), lm3d)  # 正确

        trans_params, im1, lm1, _ = align_img(face_img, lm1, self.lm3d_std)

        # vis_2dpoints(np.array(im1)[::-1, :, :], lm1)  # 正确

        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_t = torch.tensor(np.array(im1) / 255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # 放进网络中去进行系数的预测, 而且只要得到这些系数就行了, im_t是进行系数预测的, 但是lm1其实是进行loss的计算的, 在推理的时候用不上
            full_coeff = self.net_recon(im_t)
            coeffs = split_coeff(full_coeff)

        pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
        rot_mat = compute_rotation(torch.FloatTensor(pred_coeff["angle"]))

        # 这个旋转矩阵是针对y轴向上的lm, y轴向下的不适用

        # 基于这个, 所以audio2landmark得到的lm是y轴向上的, 之后还要将其转为y轴向下

        # 这里用旋转矩阵的转置, 所以得到的是转正的概率
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, torch.FloatTensor(lm_3d[np.newaxis, :, :]))[0]
        kp_rotated = np.array(kp_rotated, dtype=np.int32)

        # 通过这个可视化可以发现, 3DMM预测出来的角度转换得到旋转矩阵的转置(逆)确实可以将关键点转正, 但是这个转正实在3DMM的一个隐空间转正的, 映射不回原图上

        kp_rotated[:, 1] = 224 - 1 - kp_rotated[:, 1]
        kp_rotated = kp_rotated / trans_params[-3]

        # kp_rotated[:, :2] = kp_rotated[:, :2] + np.reshape(np.array([(336 * trans_params[-3] / 2 - 224 / 2), (336 * trans_params[-3] / 2 - 224 / 2)]), [1, 2])
        # kp_rotated[:, :2] = kp_rotated[:, :2] / trans_params[-3] - np.reshape(np.array([(336 / 2 - trans_params[-2]), (336 / 2 - trans_params[-1])]), [1, 2])  # 公式1
        # kp_rotated[:, 1] = 336 - 1 - kp_rotated[:, 1]

        # kp_rotated[:, 0] = kp_rotated[:, 0] + trans_params[-2]
        #
        # kp_rotated[:, 1] = kp_rotated[:, 1] + trans_params[-1]

        vis_2dpoints(np.array(face_img), kp_rotated)

        # _2d_vis(kp_rotated)

        pred_coeff = np.concatenate([
            pred_coeff['exp'],
            pred_coeff['angle'],
            pred_coeff['trans'],
            trans_params[2:][None],  # s, t
        ], 1)  # (1, 73)


    def run(self, frame):
        """
        Args:
            frame: bgr的图片

        Returns:
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm3d, face_img = self.face_process.detect_facelandmark(img_rgb)

        face_img = Image.fromarray(np.uint8(face_img))

        lm_3d = lm3d.copy()

        lm = lm_3d[:, :2]
        W, H = face_img.size
        lm1 = lm.reshape([-1, 2])
        lm1[:, -1] = H - 1 - lm1[:, -1]
        # vis_2dpoints(np.array(face_img), lm3d)  # 正确

        trans_params, im1, lm1, _ = align_img(face_img, lm1, self.lm3d_std)

        # vis_2dpoints(np.array(im1)[::-1, :, :], lm1)  # 正确

        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_t = torch.tensor(np.array(im1) / 255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # 放进网络中去进行系数的预测, 而且只要得到这些系数就行了, im_t是进行系数预测的, 但是lm1其实是进行loss的计算的, 在推理的时候用不上
            full_coeff = self.net_recon(im_t)
            coeffs = split_coeff(full_coeff)

        pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}

        pred_coeff["trans_params"] = trans_params[2:]

        return pred_coeff


    def frame_preprocess(self, frame):
        """
        预处理不需要有梯度, 就最为一个类似于先验信息输入的方式，插入到pipeline中去
        Args:
            frame:

        Returns:

        """
        # 不支持batch
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm3d, face_img = self.face_process.detect_facelandmark(img_rgb)  # 1. 将这个换成可以反向传播的

        # -----------------------------------------------------
        face_img = Image.fromarray(np.uint8(face_img))
        lm_3d = lm3d.copy()
        lm = lm_3d[:, :2]
        W, H = face_img.size
        lm1 = lm.reshape([-1, 2])
        lm1[:, -1] = H - 1 - lm1[:, -1]
        # vis_2dpoints(np.array(face_img), lm3d)  # 正确
        trans_params, im1, lm1, _ = align_img(face_img, lm1, self.lm3d_std)
        # vis_2dpoints(np.array(im1)[::-1, :, :], lm1)  # 正确
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        return im1, lm1, trans_params


    def frame_preprocess_tensor(self, frame_tensor):
        """
        预处理不需要有梯度, 就最为一个类似于先验信息输入的方式，插入到pipeline中去
        Args:
            frame: 是BGR的tensor格式 (H, W, C)
        Returns:

        """
        # 不支持batch
        frame = np.array(frame_tensor.detach().cpu(), dtype=np.uint8)
        # plt.imshow(frame)
        # plt.show()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm3d, _, face_bbox = self.face_process.detect_facelandmark(img_rgb, return_bbox=True)
        face_img = frame_tensor[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2], :]
        face_img = torch.nn.Upsample(size=(self.face_process.size, self.face_process.size), mode='bilinear', align_corners=True)(face_img.permute(2, 0, 1).unsqueeze(0))
        face_img = face_img[0].permute(1, 2, 0)
        # 之后用的是face_img

        # plt.imshow(frame)
        # plt.show()

        # -----------------------------------------------------
        # face_img = Image.fromarray(np.uint8(face_img))
        lm_3d = lm3d.copy()
        lm = lm_3d[:, :2]
        H, W = face_img.shape[:2]
        lm1 = lm.reshape([-1, 2])
        lm1[:, -1] = H - 1 - lm1[:, -1]
        # vis_2dpoints(np.array(face_img), lm3d)  # 正确
        trans_params, im1, lm1, _ = align_img_tensor(face_img, lm1, self.lm3d_std)

        # vis_im1 = np.array(im1[0].detach().cpu().permute(1, 2, 0), dtype=np.uint8)
        # plt.imshow(vis_im1)
        # plt.show()

        # vis_2dpoints(np.array(im1)[::-1, :, :], lm1)  # 正确
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        return im1, lm1, trans_params


    def run_coeff(self, frame):
        """

        这个就是直接返回神经网络预测的系数的
        Args:
            frame: bgr的图片
        Returns:
        """

        im1s = []
        lm1s = []
        trans_params_list = []
        if isinstance(frame, list):  # 如果是list的话, 就是有多张图片, 然后在batch的部分再cat在一起就行
            for fra in frame:
                im1, lm1, trans_params = self.frame_preprocess(fra)
                im1 = np.array(im1)[np.newaxis, :, :, :]
                im1s.append(im1)
                lm1s.append(lm1[np.newaxis, :, :])
                trans_params_list.append(trans_params[np.newaxis, :])

            im1s = np.concatenate(im1s, axis=0)
            lm1s = np.concatenate(lm1s, axis=0)
            trans_params_list = np.concatenate(trans_params_list, axis=0)

        else:
            im1, lm1, trans_params = self.frame_preprocess(frame)
            im1s = np.array(im1)[np.newaxis, :, :, :]
            lm1s = lm1[np.newaxis, :, :]
            trans_params_list = trans_params[np.newaxis, :]

        # 支持batch的
        im_t = torch.tensor(im1s / 255., dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            # 放进网络中去进行系数的预测, 而且只要得到这些系数就行了, im_t是进行系数预测的, 但是lm1其实是进行loss的计算的, 在推理的时候用不上
            full_coeff = self.net_recon(im_t)

        return full_coeff, trans_params_list, im1s

    def run_coeff_tensor(self, frame):
        """

        这个就是直接返回神经网络预测的系数的
        Args:
            frame: bgr的图片
        Returns:
        """

        im1s = []
        lm1s = []
        trans_params_list = []
        if isinstance(frame, list):  # 如果是list的话, 就是有多张图片, 然后在batch的部分再cat在一起就行
            for fra in frame:
                im1, lm1, trans_params = self.frame_preprocess_tensor(fra)
                im1s.append(im1)
                lm1s.append(lm1[np.newaxis, :, :])
                trans_params_list.append(trans_params[np.newaxis, :])

            im1s = torch.cat(im1s, dim=0)
            lm1s = np.concatenate(lm1s, axis=0)
            trans_params_list = np.concatenate(trans_params_list, axis=0)

        elif len(frame.shape) == 4 and frame.shape[0] > 1:
            for fra in frame:
                im1, lm1, trans_params = self.frame_preprocess_tensor(fra)
                im1s.append(im1)
                lm1s.append(lm1[np.newaxis, :, :])
                trans_params_list.append(trans_params[np.newaxis, :])

            im1s = torch.cat(im1s, dim=0)
            lm1s = np.concatenate(lm1s, axis=0)
            trans_params_list = np.concatenate(trans_params_list, axis=0)

        else:
            im1, lm1, trans_params = self.frame_preprocess_tensor(frame)
            lm1s = lm1[np.newaxis, :, :]
            trans_params_list = trans_params[np.newaxis, :]

        # 支持batch的
        im_t = torch.tensor(im1s / 255., dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # 放进网络中去进行系数的预测, 而且只要得到这些系数就行了, im_t是进行系数预测的, 但是lm1其实是进行loss的计算的, 在推理的时候用不上
            full_coeff = self.net_recon(im_t)

        return full_coeff, trans_params_list, im1s


class FaceReconBFM():
    def __init__(self):
        # important2: 参数化的人脸模型(BFM), 可以看看原理是不是合Flame是一样的
        self.facemodel = ParametricFaceModel(
            bfm_folder=opt_parser.bfm_folder,
            camera_distance=opt_parser.camera_d,
            focal=opt_parser.focal,
            center=opt_parser.center,
        )
        self.face_recon = FaceRecon("models/_3dmm/BFM/epoch_20.pth",
                                    "models/_3dmm/BFM")


    def run(self, frames):
        """
        Args:
            frame: frame是一张rbg的图片, 也可以是包含多张图片的list
        Returns:
        """
        coeffs, trans_params, align_im = self.face_recon.run_coeff(frames)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm, self.face_proj = self.facemodel.compute_for_render(coeffs)
        for i, _ in enumerate(self.face_proj):
            this_align_im = np.array(align_im[i], dtype=np.uint8)
            vis_2dpoints(this_align_im, self.face_proj[i])


    def run_tensor(self, images_bgr_tensors):
        """
        Args:
            frame: frame是一张rbg的图片, 也可以是包含多张图片的list
        Returns:
            投影到2d平面上的face_mesh
        """
        coeffs, trans_params, align_im = self.face_recon.run_coeff_tensor(images_bgr_tensors)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm, self.face_proj = self.facemodel.compute_for_render(coeffs)
        # for i, _ in enumerate(self.face_proj):
        #     this_align_im = np.array(align_im[i].detach().cpu().permute(1, 2, 0), dtype=np.uint8).copy()
        #     vis_2dpoints(this_align_im, np.array(self.face_proj[i].detach().cpu()))

        return self.face_proj



# if __name__ == "__main__":
#     filename = "examples/face_1.png"
#     image_bgr = cv2.imread(filename)
#
#     # face_recon = FaceRecon("D:/audio2face/SadTalker/checkpoints/epoch_20.pth",
#     #                        "D:/audio2face/SadTalker/checkpoints/BFM_Fitting")
#     # face_recon.run(image_bgr)
#
#     filename_dir = "examples/images"
#
#     images_bgr = []
#
#     names = os.listdir(filename_dir)
#
#     for name in names:
#         cur_img = cv2.imread(os.path.join(filename_dir, name))
#         images_bgr.append(cur_img)
#
#     frb = FaceReconBFM()
#     frb.run(images_bgr)


if __name__ == "__main__":
    filename_dir = "examples/images"
    images_bgr_tensors = []
    names = os.listdir(filename_dir)
    for name in names:
        # cur_img: FloatTensor的格式, shape=(H, W, C)
        cur_img = torch.FloatTensor(cv2.imread(os.path.join(filename_dir, name))).cuda()

        # plt.imshow(np.array(cur_img, dtype=np.uint8))
        # plt.show()

        images_bgr_tensors.append(cur_img)
    frb = FaceReconBFM()
    face_proj = frb.run_tensor(images_bgr_tensors)

    print(face_proj.shape)




