# coding=utf-8
# ############## Settings ###############
visualize_rate = 0.6  # 可视化倍率
yolo_rate = 0.25  # YOLO处理倍率
ID_max = 30  # ID最大为多少
person_max = 15  # 每个视角下最多能够检测的人数
epi_line_flag = False  # 是否要显示一个人的对极几何线
epi_points_cnt = 200  # 设定积累多少个公共点进行F求解
F_frames = 300  # 每隔多少帧进行一次F更新

img_save_flag = False  # 是否保存图像
img_save_path = '/home/fudan/Desktop/zls/tracking_pj/video/results/'  # 处理后的图像所存储的位置
# #######################################
import os
import sys
import numpy as np
from opt import opt
import cv2
import inference
import torch.utils.data
from dataloader import WebcamLoader, DataWriter
from fn import getTime
from yolo.preprocess import prep_frame
# #############################
from src.init import init_frames  # 初始化器
from src.tools import load_data  # 正常帧加载器
from src.tools import Visualize  # 可视化过程，其中含有路径配置
from src.match import Hungarian_match, Feature_match, Update_tracker  # 单通道匹配函数
from src.match import Inter_cam_match_1, Inter_cam_match_2
from src.tools import Get_F # , save_img
import time
from torch.autograd import Variable

# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
cam_number = 2
match_id_cam, match_id = [], []  # 记录五帧内帧间关联信息（由mvpose结果进行联合分析）
init_flag = 1  # 是否在初始化追踪器期间，默认处于
init_info = [[], [], [], []]  # 对初始化25帧进行记录
# #############################

args = opt


def loop():
    n = 0
    while True:
        yield n
        n += 1


def resize_vis(img_0):
    return cv2.resize(img_0, (int(img_0.shape[1] * visualize_rate), int(img_0.shape[0] * visualize_rate)), interpolation=cv2.INTER_AREA)


def resize_yolo(img_0):
    return cv2.resize(img_0, (int(img_0.shape[1] * 0.25), int(img_0.shape[0] * 0.25)), interpolation=cv2.INTER_AREA)


def preprocess(frame_0, frame_1):
    frame = np.concatenate([frame_0, frame_1], 0)
    inp_dim = int(args.inp_dim)  # default=608
    img, orig_img, dim = prep_frame(frame, inp_dim)
    im_dim_list = torch.FloatTensor([dim]).repeat(1, 2)
    return img, orig_img, dim, im_dim_list


if __name__ == '__main__':

    url_1 = "rtsp://linye:linye123@192.168.200.253:554/Streaming/Channels/101"
    url_2 = "rtsp://linye:linye123@192.168.200.253:554/Streaming/Channels/301"
    num_cam = 2
    webcam = args.webcam
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load input video
    fvs_0 = WebcamLoader(url_1).start()
    fvs_1 = WebcamLoader(url_2).start()

    # detection module
    print('Loading detection model ')
    sys.stdout.flush()
    det_model = inference.yolo_detecter()

    # pose module
    print('Loading pose model')
    sys.stdout.flush()
    pose_model = inference.pose_detection()

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = loop()  # tqdm(loop())
    for i in im_names_desc:
        try:
            start_time = getTime()
            begin = time.time()

            # ##################################  Get Frames  ####################################
            print('\n******************* Frame:%d ********************' % i)
            img_0 = fvs_0.read()
            img_1 = fvs_1.read()
            # 可视化显示变成1/2
            fvis_0 = resize_vis(img_0)
            fvis_1 = resize_vis(img_1)
            # YOLO处理图片时变成1/4
            frame_0 = resize_yolo(img_0)
            frame_1 = resize_yolo(img_1)
            single_height = frame_0.shape[0]  # print(frame_0.shape) # (432, 768, 3)

            # ##################################  Pre Process  ####################################
            img, orig_img, dim, im_dim_list = preprocess(frame_0, frame_1)

            # ##################################  Detection  ####################################
            with torch.no_grad():
                img = Variable(img).cuda()
                im_dim_list = im_dim_list.cuda()
                # ###################
                yolo_start = time.time()
                prediction = det_model.get_prediction(img, cuda=True)
                yolo_end = time.time()
                _yolo_delta = yolo_end - yolo_start
                print('YOLO_time : %f' % (1 / _yolo_delta))
                # ###################
                flag, boxes, scores = det_model.NMS_process(prediction, i, fvis_0, fvis_1, im_dim_list)
                if flag == False:
                    # save_img(i, cam_number, img_save_flag, img_save_path, [fvis_0, fvis_1])
                    continue
                # pose estimation
                flag, hm = pose_model.get_prediction(orig_img, boxes, scores, single_height, person_max)
                if flag == True:
                    (box, box_s, roi, kp, kp_s) = hm

                print('POSE_time : %f' % (1 / (time.time() - yolo_end)))
                # ###################

            # ##################################  matching  ####################################
            frame_id, bbox, bbox_score, imgs, keyp, keyp_s = i, box, box_s, roi, kp, kp_s

            # 处于初始化期间
            if init_flag == 1:
                init_info[0].append(frame_id)
                init_info[1].append(bbox)
                init_info[2].append(bbox_score)
                init_info[3].append(imgs)
                if len(init_info[0]) != 25:  # 不满25帧，结束当前循环，进行积累
                    continue
                else:
                    # 初始化追踪器 tracker / tracker_id
                    init_flag, Frames_Lib, tracker, tracker_id, tracker_cnt, feature_model = init_frames(cam_number, init_info)
                    init_info = [[], [], [], []]  # 记录器清零
                    same_points, F = [[], []], []  # 自标定F的公共点
                    continue

            # 处于正常处理状态
            elif init_flag == 0:

                # 获取当前时刻roi信息,对检测结果进行过滤
                rois, ids, features, imgs, score_cam, keyp, keyp_s = load_data(cam_number, [visualize_rate, yolo_rate], bbox, bbox_score, imgs, keyp, keyp_s, feature_model)

                # 单通道追踪更新，根据匈牙利匹配算法对IoU进行匹配，得到ID列表
                ID_list_cam = Hungarian_match(rois, cam_number, tracker, tracker_id)

                # 使用25f法对IoU方法进行修正，时间域Mvpose
                ID_list_cam = Feature_match(cam_number, Frames_Lib, ID_list_cam, rois, features)

                # 对于25F法也处理不了的IoU，只能新添加ID
                Update_tracker(cam_number, Frames_Lib, ID_list_cam, rois, features, tracker, tracker_id, ID_max)

                # 多通道联合分析（首先生成ID链接信息，后生成总的ID）
                match_id_cam, match_id = Inter_cam_match_1(cam_number, frame_id, ID_list_cam, match_id_cam, match_id, features, keyp, F)  # 获取
                fuse_ID = Inter_cam_match_2(tracker, tracker_id, match_id_cam, match_id)

                # Bbox可视化，keypoints可视化，对极几何线可视化
                Visualize([fvis_0, fvis_1], cam_number, frame_id, img_save_flag, img_save_path, tracker, tracker_id, fuse_ID, keyp, keyp_s, F, epi_line_flag)

                # 进行F求解
                if len(F) == 0 or frame_id % F_frames == 0:  # 每隔固定帧进行一次F重标定
                    F = []
                    same_points, F = Get_F(same_points, fuse_ID, keyp, keyp_s, epi_points_cnt)  # 进行自标定

            end_time = time.time()
            print('***** FPS: %f \n' % (1 / (end_time - begin)))

        except KeyboardInterrupt:
            break
