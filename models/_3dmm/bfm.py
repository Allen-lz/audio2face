"""This script defines the parametric 3d face model for Deep3DFaceRecon_pytorch
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import os
import matplotlib.pyplot as plt
from array import array
import os.path as osp

# load expression basis
def LoadExpBasis(bfm_folder='BFM'):
    n_vertex = 53215
    Expbin = open(osp.join(bfm_folder, 'Exp_Pca.bin'), 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(Expbin, 1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(Expbin, 3*n_vertex)
    expPC.fromfile(Expbin, 3*exp_dim[0]*n_vertex)
    Expbin.close()

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(osp.join(bfm_folder, 'std_exp.txt'))

    return expPC, expEV

# transfer original BFM09 to our face model
def transferBFM09(bfm_folder='BFM'):
    print('Transfer BFM09 to BFM_model_front......')
    original_BFM = loadmat(osp.join(bfm_folder, '01_MorphableModel.mat'))
    shapePC = original_BFM['shapePC']  # shape basis
    shapeEV = original_BFM['shapeEV']  # corresponding eigen value
    shapeMU = original_BFM['shapeMU']  # mean face
    texPC = original_BFM['texPC']  # texture basis
    texEV = original_BFM['texEV']  # eigen value
    texMU = original_BFM['texMU']  # mean texture

    expPC, expEV = LoadExpBasis()

    # transfer BFM09 to our face model

    idBase = shapePC*np.reshape(shapeEV, [-1, 199])
    idBase = idBase/1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC*np.reshape(expEV, [-1, 79])
    exBase = exBase/1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC*np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # our face model is cropped along face landmarks and contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    index_exp = loadmat(osp.join(bfm_folder, 'BFM_front_idx.mat'))
    index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat(osp.join(bfm_folder, 'BFM_exp_idx.mat'))
    index_shape = index_shape['trimIndex'].astype(
        np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3])/1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.
    other_info = loadmat(osp.join(bfm_folder, 'facemodel_info.mat'))
    frontmask2_idx = other_info['frontmask2_idx']
    skinmask = other_info['skinmask']
    keypoints = other_info['keypoints']
    point_buf = other_info['point_buf']
    tri = other_info['tri']
    tri_mask2 = other_info['tri_mask2']

    # save our face model
    savemat(osp.join(bfm_folder, 'BFM_model_front.mat'), {'meanshape': meanshape, 'meantex': meantex, 'idBase': idBase, 'exBase': exBase, 'texBase': texBase,
            'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2, 'keypoints': keypoints, 'frontmask2_idx': frontmask2_idx, 'skinmask': skinmask})

def _3d_vis(points):
    points = points[0]

    # print("points.shape=", points.shape)

    points = np.array(points.detach().cpu())
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2], zdir='z', c='c')
    plt.show()


def _2d_vis(points):

    points = np.array(points.cpu().detach(), dtype=np.int32)

    if len(points.shape) == 3:
        points = points[0]
    p1 = points[:, 0]  # 数据点
    p2 = points[:, 1]
    # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
    plt.figure('Draw')

    print(p1.shape)
    print(p2.shape)

    plt.scatter(p1, p2)  # scatter绘制散点图
    # plt.draw()  # 显示绘图
    plt.show()

def perspective_projection(focal, center):
    """
    Args:
        focal:    焦距
        center:  成像平面的中心

    Returns:

    """
    # return p.T (N, 3) @ (3, 3)

    # 这个是相机的内参K, 用来投影并转像素坐标的
    # 这里默认的dx和dy等于1
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]



class ParametricFaceModel:
    def __init__(self, 
                bfm_folder='./BFM', 
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,
                center=112.,  # 成像屏幕的大小是224, 中心点固然就是112
                is_train=False,
                default_name='BFM_model_front.mat', device="cuda"):
        
        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            transferBFM09(bfm_folder)
        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3*N, 1] T-Pose
        self.mean_shape = model['meanshape'].astype(np.float32)
        # identity basis. [3*N, 80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*N, 64]
        self.exp_base = model['exBase'].astype(np.float32)

        # mean face texture. [3*N, 1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)
        # texture basis. [3*N, 80]
        self.tex_base = model['texBase'].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N, 8], 这个8我是没有搞清楚
        # 这个是每个顶点的索引, 索引从0开始
        self.point_buf = model['point_buf'].astype(np.int64) - 1

        # vertex indices for each face. starts from 0. [F,3], 这个应该是脸上的一片一片的三角形, 三角形切面, 用于求法向量的
        self.face_buf = model['tri'].astype(np.int64) - 1  # 三角形切面:  (70789, 3)
        # vertex indices for 68 landmarks. starts from 0. [68,1], 从几千个点中检索68个关键点的index
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if is_train:
            # vertex indices for small face region to compute photometric error. starts from 0.
            self.front_mask = np.squeeze(model['frontmask2_idx']).astype(np.int64) - 1  # (27660, )
            # vertex indices for each face from small face region. starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1  # (54681, 3)  正脸的三角形切面
            # vertex indices for pre-defined skin region to compute reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])  # (35709,)这个是皮肤的index
        
        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        # 透视投影, 将相机坐标下的顶点投影到成像平面上
        self.persc_proj = perspective_projection(focal,
                                                 center  # 这里有个与尺度相关的参数
                                                 )  # 但是这里也只有缩放和平移


        self.device = device
        self.camera_distance = camera_distance  # 相机距离
        self.SH = SH()  # 光照的我不看
        self.init_lit = torch.FloatTensor(init_lit.reshape([1, 1, -1]).astype(np.float32))
        

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    
    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs

        就是简单的矩阵相乘加权(id + exp)
        """
        batch_size = id_coeff.shape[0]



        id_part = torch.einsum('ij,aj->ai', torch.FloatTensor(self.id_base).to(self.device), id_coeff)
        exp_part = torch.einsum('ij,aj->ai', torch.FloatTensor(self.exp_base).to(self.device), exp_coeff)
        face_shape = id_part + exp_part + torch.FloatTensor(self.mean_shape.reshape([1, -1])).to(self.device)
        return face_shape.reshape([batch_size, -1, 3])
    

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', torch.FloatTensor(self.tex_base).to(self.device), tex_coeff) + torch.FloatTensor(self.mean_tex).to(self.device)
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, -1, 3])


    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        # 得到三角形的三个顶点, 计算两个向量
        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3

        # 通过叉乘来计算三角形切面的法向量
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)  # 使用L2范数的归一化  [1, 70789, 3]

        # face_norm.shape, torch.zeros(face_norm.shape[0], 1, 3).to(self.device).shape = torch.Size([1, 70789, 3]) torch.Size([1, 1, 3])
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)  # torch.Size([1, 70790, 3])

        # [1, 70789, 3] with index [35709, 8] --> [1, 35709, 8, 3]
        # 这里的 35709 * 8都是针对70789的index, 但是最后的矩阵要reshape成(35709 * 8)
        # face_norm[:, self.point_buf].shape = torch.Size([1, 35709, 8, 3])
        # self.point_buf.shape = torch.Size([35709, 8])

        # 这里不是很懂为什么把8个点的向量都加起来作为一个点的法向量, 其实当前的目标向量算是一个中心向量吗, 所以要计算一个向量的加权
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)  # 最后再求一个二范数
        return vertex_norm


    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])

        gamma = gamma + self.init_lit.to(self.device)
        gamma = gamma.permute(0, 2, 1)



        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    
    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        # x, y, z是每个人脸相对于每个轴的正方向的角度

        # 绕着x轴进行旋转, 所以x轴是不懂的
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
        return rot.permute(0, 2, 1)


    def to_camera(self, face_shape):
        """
        这里我猜的是, 相机在一个远方的某个点位上, 算是一个参考点,
        Args:
            face_shape:

        Returns:

        """
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        # 进一个缩放平移, 但是这里为什么是右乘, 意思是对face_shape的坐标系进行旋转平移吗

        # persc_proj中平移的部分用的是[center, center]

        # if isinstance(face_shape, torch.Tensor):
        #     face_shape = np.array(face_shape.cpu().detach())

        face_proj = face_shape @ torch.FloatTensor(self.persc_proj).to(face_shape.device)
        # 这里除以一个depth, 投影到平面上(这个投影还保存了深度的信息)
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        # _2d_vis(face_proj)
        # assert False
        return face_proj

    def to_image_tensor(self, face_shape):
        """

        之前的函数需要将face_shape转成numpy格式的, 这里我统一使用
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        # 进一个缩放平移, 但是这里为什么是右乘, 意思是对face_shape的坐标系进行旋转平移吗

        # persc_proj中平移的部分用的是[center, center]

        face_proj = torch.matmul(face_shape, torch.FloatTensor(self.persc_proj).to(face_shape.device))
        # 这里除以一个depth, 投影到平面上(这个投影还保存了深度的信息)
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        # _2d_vis(face_proj)
        # assert False
        return face_proj


    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)


    def get_landmarks(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)

        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """  
        return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
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
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }
    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)

            在这个步骤计算的时候, 我没有看到使用固定size的
        """
        coef_dict = self.split_coeff(coeffs)
        # id    torch.Size([1, 80])
        # exp   torch.Size([1, 64])
        # tex   torch.Size([1, 80])
        # angle torch.Size([1, 3])  # 这个是旋转轴角
        # gamma torch.Size([1, 27])
        # trans torch.Size([1, 3])  # 这个是三个轴方向的偏移

        # 一开始这个应该是slam的世界坐标, 就是3DMM预设的那个坐标
        face_shape = self.compute_shape(coef_dict['id'],
                                        coef_dict['exp']
                                        )  # torch.Size([1, 35709, 3])
        # _3d_vis(face_shape)

        rotation = self.compute_rotation(coef_dict['angle'])  # 将轴角转为旋转矩阵, 对重建出来的顶点进行旋转, 还原真实的头部姿态
        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        # _3d_vis(face_shape_transformed)

        # 为face_shape_transformed添加上相机的距离
        """
        我感觉需要在这里对face_vertex进行一个缩放, face_vertex被进行缩放之后也会影响到后面的一些结果的产生 
        """
        face_vertex = self.to_camera(face_shape_transformed)
        # _3d_vis(face_vertex)

        # 这里投影的时候会涉及到缩放呢
        face_proj = self.to_image(face_vertex)   # mesh的中心点被移动到了(center, center)上, 并且点投影到了图像的平面上

        face_proj_np = np.array(face_proj.detach().cpu())

        # _2d_vis(np.array(face_proj.detach().cpu()[0]))

        # 从投影之后的点中取出68个点
        landmark = self.get_landmarks(face_proj_np)  # (b, 68, 2)
        # ================================= 这里针对shape的操作就到此为止了 ====================================

        # 利用估计出来的参数进行加权得到偏移
        face_texture = self.compute_texture(coef_dict['tex'])
        # 基于三角形切面的索引, 以及顶点, 进行法线的计算, 这里是计算每个切面的法线而不是每个顶点的法线
        #         # 注意这里使用的是最原初的法向量
        face_norm = self.compute_norm(face_shape)
        # torch.Size([b, 35709, 3])
        face_norm_roted = face_norm @ rotation  # 再对得到的法线向量进行一个旋转, 得到当前测试图片的脸部姿态

        # 渲染得到面皮, 这个应该也是render部分的, 但是我没有看
        face_color = self.compute_color(face_texture,   # 面部材质
                                        face_norm_roted,  # 法线向量
                                        coef_dict['gamma'])  # [b, 35709, 3]
        # plt.imshow(face_color)
        return face_vertex, face_texture, face_color, landmark, face_proj
