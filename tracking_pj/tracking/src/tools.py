import cv2
import numpy as np
import json
from PIL import Image
import time
import torch


def load_data(cam_number, rate, bbox, bbox_score, imgs, keyp, keyp_s, feature_model):
    """加载当前帧数据，同时进行卷积"""

    def roi_th_reid(cam, bbox, bbox_score, imgs, keyp, keyp_s, th=0.3):  # 单帧数据
        if len(bbox[cam]) == 0:
            return [], [], [], [], [], []
        else:
            roi, id, img, k, k_s = [], [], [], [], []
            score = []
            # 存入时顺便扩大keypoints和Bbox
            for roi_id in range(len(bbox[cam])):
                if bbox_score[cam][roi_id] > th:  # 有效检测
                    roi.append([int(i * (rate[0] / rate[1])) for i in bbox[cam][roi_id]])
                    id.append(roi_id)
                    score.append(bbox_score[cam][roi_id])
                    k.append((rate[0] / rate[1]) * np.array(keyp[cam][roi_id]))
                    k_s.append(keyp_s[cam][roi_id])
                    # 将imgs读进去
                    img.append(imgs[cam][roi_id])
            return roi, id, img, score, k, k_s

    def get_feature(img):
        """将所有视角下的RoI拼在一起后统一进行卷积操作"""
        roi_imgs = []
        for cam in range(len(img)):
            roi_imgs.extend(img[cam])

        if len(roi_imgs) != 0:  # 摄像头中总人数>0
            feature = feature_model.get_feature(roi_imgs)  # 对所有视角下的RoI进行卷积
        else:  # 摄像头中没人，直接返回[[],[],[]]
            return img

        cnt = 0  # 记录个数，重构feature
        feature_ = []
        for cam in range(len(img)):
            _ = []
            for i in range(len(img[cam])):
                _.append(feature[cnt])
                cnt = cnt + 1
            feature_.append(_)
        return feature_

    roi_cam, id_cam, feature_cam, img_cam, keyp_cam, keyp_s_cam = [], [], [], [], [], []
    score_cam = []
    # 读取每帧信息
    for cam in range(cam_number):
        roi, id, img, score, k, k_s = roi_th_reid(cam, bbox, bbox_score, imgs, keyp, keyp_s, th=0.3)
        score_cam.append(score)
        roi_cam.append(roi)  # roi包含当前时刻3个通道下的信息
        id_cam.append(id)
        keyp_cam.append(k)
        keyp_s_cam.append(k_s)
        img_cam.append(img)

    feature_cam = get_feature(img_cam)

    return roi_cam, id_cam, feature_cam, img_cam, score_cam, keyp_cam, keyp_s_cam


def keypoint_vis(frame, keyp, keyp_s, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PURPLE = (255, 0, 255)

    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16), (5, 11), (6, 12), (11, 12)]

        p_color = [(255, 255, 0), (255, 0, 255), (0, 255, 255),
                   (255, 0, 0), (0, 255, 0), (0, 0, 255),
                   (255, 192, 203), (75, 0, 130), (0, 191, 255),
                   (127, 255, 212), (128, 128, 0), (255, 215, 0),
                   (205, 133, 63), (139, 69, 19), (255, 99, 71),
                   (250, 128, 114), (178, 34, 34)]

        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                      (0, 0, 255), (255, 0, 0), (0, 165, 255)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]

    for id in range(len(keyp)):
        part_line = {}
        kp_preds = keyp[id]
        kp_scores = keyp_s[id]

        # Draw keypoints
        for n in range(len(kp_scores)):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(frame, start_xy, end_xy, line_color[i], 2)  # 2 * (kp_scores[start_p] + kp_scores[end_p]) + 1
    return frame


def epi_line_vis(cam, frame, F, keyp):
    def drawlines(img, lines):
        colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203), (75, 0, 130), (0, 191, 255),
                  (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19), (255, 99, 71), (250, 128, 114), (178, 34, 34)]
        r, c, _ = img.shape
        for id, r in enumerate(lines):
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img = cv2.line(img, (x0, y0), (x1, y1), colors[id], 1)
        return img

    # 无F或者不是第二通道图像的话，直接返回原帧
    if len(F) == 0 or cam == 0 or len(keyp[0]) == 0:
        return frame

    lines_01 = cv2.computeCorrespondEpilines(np.array(keyp[0][0]).astype(int).reshape(-1, 1, 2), 2, F[1])
    lines_01 = lines_01.reshape(-1, 3)

    frame = drawlines(frame, lines_01)
    return frame


def Visualize(video_frame, cam_number, frame_id, img_save_flag, img_save_path, tracker, tracker_id, tracker_id_, keyp, keyp_s, F, epi_line_flag):
    hi, we = video_frame[0].shape[0], video_frame[0].shape[1]

    library_colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
                      (150, 0, 255), (0, 191, 255), (127, 255, 212), (128, 255, 0), (255, 215, 0), (205, 133, 63), (255, 69, 19),
                      (255, 99, 71), (250, 128, 114), (178, 34, 34), (255, 128, 0), (255, 0, 255), (32, 178, 170), (0, 139, 139),
                      (70, 130, 180), (30, 144, 255)]

    blank_image = np.zeros((hi, we * cam_number, 3), np.uint8)

    for cam in range(0, cam_number):  # 根据ID完成可视化
        for i in range(len(tracker[cam])):  # 遍历每一个RoI
            roi = tracker[cam][i]
            ID = tracker_id_[cam][i]
            # 添加ID和Bbox框
            cv2.putText(video_frame[cam], 'ID:' + str(ID), (roi[0], roi[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, library_colors[ID % 23], 2)
            cv2.rectangle(video_frame[cam], (roi[0], roi[1]), (roi[2], roi[3]), library_colors[ID % 23], 2)
            # 绘制一组对极几何线
            if epi_line_flag:
                video_frame[cam] = epi_line_vis(cam, video_frame[cam], F, keyp)

        video_frame[cam] = keypoint_vis(video_frame[cam], keyp[cam], keyp_s[cam])
        blank_image[:, cam * we:(cam + 1) * we, :] = video_frame[cam]

    cv2.namedWindow('MCMT', cv2.WINDOW_NORMAL)  # , cv2.WINDOW_AUTOSIZE
    cv2.imshow('MCMT', blank_image)
    cv2.waitKey(1)
    # if img_save_flag:
    #     cv2.imwrite(img_save_path + str(frame_id) + '.jpg', blank_image)


def save_img(frame_id, cam_number, img_save_flag, img_save_path, video_frame):
    hi, we = video_frame[0].shape[0], video_frame[0].shape[1]
    blank_image = np.zeros((hi, we * cam_number, 3), np.uint8)
    for cam in range(0, cam_number):  # 根据ID完成可视化
        blank_image[:, cam * we:(cam + 1) * we, :] = video_frame[cam]
    cv2.namedWindow('show1')  # , cv2.WINDOW_AUTOSIZE
    cv2.imshow('show1', blank_image)
    cv2.waitKey(1)
    if img_save_flag:
        cv2.imwrite(img_save_path + str(frame_id) + '.jpg', blank_image)


def Get_F(same_points, fuse_ID, keyp, keyp_s, points_cnt):
    if len(fuse_ID) == 0:
        return same_points, []
    for id_0 in range(len(fuse_ID[0])):  # 遍历摄像头1中的ID
        if fuse_ID[0][id_0] in fuse_ID[1]:  # 出现公共ID
            id_1 = fuse_ID[1].index(fuse_ID[0][id_0])  # 确定了同人的下标索引
            for i in range(17):
                if keyp_s[0][id_0][i] > 0.8 and keyp_s[1][id_1][i] > 0.8:  # 取置信度大于0.5的点
                    same_points[0].append(tuple(keyp[0][id_0][i]))
                    same_points[1].append(tuple(keyp[1][id_1][i]))
    if len(same_points[0]) > points_cnt:  # 总点数大于200
        F_0_1, _ = cv2.findFundamentalMat(np.array(same_points[0]), np.array(same_points[1]), cv2.FM_RANSAC)
        F_1_0, _ = cv2.findFundamentalMat(np.array(same_points[1]), np.array(same_points[0]), cv2.FM_RANSAC)
        print('Get F Mat:', [F_0_1, F_1_0])
        return [[], []], [F_0_1, F_1_0]
    else:
        return same_points, []
