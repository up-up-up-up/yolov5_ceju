import cv2
import numpy as np


# ************************************Little Stereo Param_CalibByHand*****************************************
left_camera_matrix = np.array([[1101.89299, 0, 1119.89634],
                               [0, 1100.75252, 636.75282],
                               [0, 0, 1]])

left_distortion = np.array([[-0.08369, 0.05367, -0.00138, -0.00090, 0.00000]])

right_camera_matrix = np.array([[1091.11026, 0, 1117.16592],
                                [0, 1090.53772, 633.28256],
                                [0, 0, 1]])

right_distortion = np.array([[-0.09585, 0.07391, -0.00065, -0.00083, 0.00000]])

R = np.matrix([
    [1.0000, -0.000603116945856524, 0.00377055351856816],
    [0.000608108737333211, 1.0000, -0.00132288199083992],
    [-0.00376975166958581, 0.00132516525298933, 1.0000],
])

print(R)

T = np.array([ -119.99423, -0.22807, 0.18540])  # 平移关系向量

size = (2280, 1242)  # 图像尺寸


# ************************************************************************************************************

# *****************************************Little Stereo Param MatlabAuto***************************************************
# left_camera_matrix = np.array([[490.7535, -0.0117, 310.7994],
#                                [0, 490.8401, 262.6784],
#                                [0, 0, 1]])
#
# left_distortion = np.array([[0.0613, 0.1157, 0.00029, 0.00018, -0.3991]])
#
# right_camera_matrix = np.array([[492.5555, -0.1940, 331.4132],
#                                 [0, 492.6360, 241.9927],
#                                 [0, 0, 1]])
#
# right_distortion = np.array([[0.0521, 0.1648, 0.00020, 0.00025, -0.4944]])
#
# R = np.matrix([
#     [1.0000, -0.00060, 0.0038],
#     [0.00059, 1.0000, 0.0013],
#     [-0.0038, -0.0013, 1.0000],
# ])
#
# print(R)
#
# T = np.array([-99.2088, -0.1603, -0.1892])  # 平移关系向量
#
# size = (640, 480)  # 图像尺寸




# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
# left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
# right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_32FC1)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_32FC1)
