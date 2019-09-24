import cv2
import numpy as np
import json
from PIL import Image
import time


def load_data(cam_number, frame_id, bbox, bbox_score, imgs, feature_model):
    """加载当前帧数据，同时进行卷积"""

    def roi_th_reid(cam, bbox, bbox_score, imgs, th=0.3):  # 单帧数据
        if len(bbox[cam]) == 0:
            return [], [], [],[]
        else:
            roi, id,  img = [], [], []
            score = []
            for roi_id in range(len(bbox[cam])):
                if bbox_score[cam][roi_id] > th:  # 有效检测
                    roi.append(bbox[cam][roi_id])
                    id.append(roi_id)
                    score.append(bbox_score[cam][roi_id])
                    # 将imgs读进去
                    img.append(imgs[cam][roi_id])
            return roi, id, img,score

    def get_feature(img):
        """将所有视角下的RoI拼在一起后统一进行卷积操作"""
        roi_imgs = []
        for cam in range(len(img)):
            roi_imgs.extend(img[cam])

        if len(roi_imgs) != 0:  # 摄像头中总人数>0
            feature = feature_model.get_feature(roi_imgs)  # 对所有视角下的RoI进行卷积
        else:   # 摄像头中没人，直接返回[[],[],[]]
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

    roi_cam, id_cam, feature_cam, img_cam = [], [], [], []
    score_cam = []
    # 读取每帧信息
    for cam in range(cam_number):
        roi, id, img, score = roi_th_reid(cam, bbox, bbox_score, imgs, th=0.3)
        score_cam.append(score)
        roi_cam.append(roi)  # roi包含当前时刻3个通道下的信息
        id_cam.append(id)
        img_cam.append(img)

    feature_cam = get_feature(img_cam)

    return roi_cam, id_cam, feature_cam, img_cam,score_cam


def Visualize(video_frame,cam_number, frame_id, tracker, tracker_id, tracker_id_):
    # video_frame_paths = ['/home/fudan/Desktop/hs/alpha_pose/test_1000/cam1/',
    #                      '/home/fudan/Desktop/hs/alpha_pose/test_1000/cam2/',
    #                      '/home/fudan/Desktop/hs/alpha_pose/test_1000/cam3/']
    library_colors = [
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 192, 203),
        (75, 0, 130), (0, 191, 255), (127, 255, 212), (128, 128, 0), (255, 215, 0), (205, 133, 63), (139, 69, 19),
        (255, 99, 71), (250, 128, 114), (178, 34, 34), (128, 0, 0), (222, 184, 135), (32, 178, 170), (0, 139, 139),
        (70, 130, 180), (30, 144, 255),
        (255, 255, 255),
    ]

    blank_image = np.zeros((144, 256 * cam_number, 3), np.uint8)
    for cam in range(0, cam_number):  # 根据ID完成可视化
        #video_frame_path = video_frame_paths[cam] + '%04d' % (frame_id) + '.jpg'  # 载入原图像
        #video_frame = cv2.imread(video_frame_path)  # 获取到原图像

        for i in range(len(tracker[cam])):  # 遍历每一个RoI
            roi = tracker[cam][i]
            ID = tracker_id_[cam][i]
            cv2.putText(video_frame[cam], 'ID:' + str(ID), (int(roi[0]), int(roi[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        library_colors[ID],2)
            cv2.rectangle(video_frame[cam], (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), library_colors[ID], 3)

        blank_image[:, cam * 256:(cam + 1) * 256, :] = video_frame[cam]

    #cv2.imwrite('G:/pose-estimation/tracking/result/webcam_output/' + str(frame_id) + '.jpg', blank_image)
    cv2.namedWindow('show', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('show',blank_image)
    cv2.waitKey(30)

def get_video(start_frame, end_frame):
    
    import cv2
    videoWriter = cv2.VideoWriter('0.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (3840 * 3, 2160))  # 读入视频文件
    for frame_id in range(start_frame, end_frame):
        print(frame_id)
        img = cv2.imread('/home/fudan/Desktop/wsl/Track_Recon/results/' + str(frame_id) + '.jpg')
        videoWriter.write(img)


def YOLO(cam_number, frame):
    frame_id, bbox, bbox_score, img = frame, [], [], []

    for cam in range(cam_number):
        frame_data = json.load(  # 获取三个摄像头信息
            open('/home/fudan/Desktop/hs/alpha_pose/res_1000/cam' + str(cam + 1) + '/' + '%04d' % (
                frame_id) + '/' + 'bbox_keypoints.json', 'r'))
        bbox.append([frame_data[i]["bboxes"] for i in range(len(frame_data))])
        bbox_score.append([frame_data[i]["bbox_score"][0] for i in range(len(frame_data))])
        img.append([Image.open(
            '/home/fudan/Desktop/hs/alpha_pose/res_1000/' + 'cam' + str(cam + 1) + '/' + '%04d' % (
                frame_id) + '/' + 'rois' + '/' + str(i) + '.jpg').convert('RGB') for i in range(len(frame_data))])
    for cam in range(len(img)):
        for i in range(len(img[cam])):
            img[cam][i].resize((100, 200))
    return frame_id, bbox, bbox_score, img
