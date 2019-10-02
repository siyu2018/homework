import cv2
import os
import numpy as np

imageNum = 20
squareSize = 28  # 单个棋盘格大小，单位mm
boardSize = (6, 8)  # 网格内角点数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)
imagePath = './image'
for name in os.listdir(imagePath):
    img = cv2.imread(imagePath + '/' + name, 0)
    img = cv2.resize(img, (1280, 720))
    ret, corners = cv2.findChessboardCorners(img, boardSize, None)
    if ret == True:
        cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, boardSize, corners, ret)
        # cv2.imshow('findCorners', img)
        # cv2.waitKey(1000)
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
img2 = cv2.imread('./image/6.jpg')
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

print("ret:", ret)
print("mtx:\n", mtx)  # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs)  # 平移向量  # 外参数

cv2.imshow('src', img2)
cv2.imshow('calib_result', dst)
cv2.waitKey(0)
