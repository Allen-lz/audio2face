"""
错误: SystemError: CPUDispatcher(<function _get_preds_fromhm at 0x0000024689C593A8>) returned a result with an error set
参考博客: https://github.com/1adrianb/face-alignment/issues/288
解决方案: uninstalled numpy-1.21.6 and install numpy-1.18.0

"""
import face_alignment
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Face_Align():
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    def __run__(self, img_rgb):

        plt.imshow(img_rgb)
        plt.show()


        preds = self.fa.get_landmarks(img_rgb)[0]
        kps = np.array(preds, dtype=np.int32)
        return kps

    def _3d_vis(self, points):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2], zdir='z', c='c')
        plt.show()

    def vis_2dpoints(self, img, points):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for p in points:

            cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 255, 0), -1)
        plt.imshow(img)
        plt.show()

    def drawPart(self, face_kps,drawimg,start,end,color,closed=True):
        for i in range(start, end+1):
            cv2.circle(drawimg, (face_kps[i,0],face_kps[i,1]),2,color,thickness=1)
            if i < end:
                cv2.line(drawimg,(face_kps[i,0],face_kps[i,1]),(face_kps[i+1,0],face_kps[i+1,1]),color,thickness=2)
            elif closed:
                cv2.line(drawimg,(face_kps[end,0],face_kps[end,1]),(face_kps[start,0],face_kps[start,1]),color,thickness=2)
        return drawimg

    def drawFace(self, kps, img):
        # 在绘制之前, 先将68点的形式转成blendershape的形式
        kps = self.landmark2blendershape(kps)
        img = self.drawPart(kps, img, 0, 4, (255, 0, 0), False)  # 左眉毛，非封闭区域
        img = self.drawPart(kps, img, 5, 9, (0, 255, 0), False)  # 右眉毛，非封闭区域
        img = self.drawPart(kps, img, 10, 15, (255, 0, 0))  # 左眼
        img = self.drawPart(kps, img, 16, 21, (0, 255, 0))  # 右眼
        img = self.drawPart(kps, img, 22, 25, (0, 0, 255), False)  # 鼻梁，非封闭区域
        img = self.drawPart(kps, img, 26, 30, (0, 0, 255), False)  # 鼻子，非封闭区域
        img = self.drawPart(kps, img, 31, 42, (0, 255, 255))  # 外嘴唇
        img = self.drawPart(kps, img, 43, 50, (255, 0, 255))  # 内嘴唇
        img = self.drawPart(kps, img, 51, 67, (255, 255, 0), False)  # 脸轮廓，非封闭区域
        return img

    def landmark2blendershape(self, res):
        """
        0-16 --> 51-67
        17-21 --> 0-4
        22-26 --> 5-9
        27-30 --> 22-25
        31-35 --> 26-30
        36-41 --> 10-15
        42-47 --> 16-21
        48-67 --> 31-50
        :return:
        """
        new_res = np.zeros(res.shape)
        new_res[51: 68] = res[0: 17]
        new_res[0: 5] = res[17: 22]
        new_res[5: 10] = res[22: 27]

        new_res[22: 26] = res[27: 31]
        new_res[26: 31] = res[31: 36]

        new_res[10: 16] = res[36: 42]

        new_res[16: 22] = res[42: 48]
        new_res[31: 51] = res[48: 68]

        new_res = np.array(new_res, dtype=np.int32)

        return new_res

if __name__ == "__main__":

    image_path = 'E:/styletransfer/VToonify/images/20230319124219.png'

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    facealignment = Face_Align()
    kps = facealignment.__run__(img_rgb)

    img_rgb = facealignment.drawFace(kps, img_rgb)

    plt.imshow(img_rgb)
    plt.show()




