import os
import argparse
import os.path as osp
import pickle
import sys
import numpy as np
from tqdm import tqdm
from opt import opt
from PIL import Image
import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
from dataloader import WebcamLoader, DataWriter, crop_from_dets, Mscoco,PoseLoader,DetectionLoader,FeatureLoader,IOULoader
from yolo.darknet import Darknet
# from reid import reid_interface
from yolo.preprocess import prep_frame
from SPPE.src.main_fast_inference import *
from yolo.util import write_results, dynamic_write_results
from pPose_nms import pose_nms
from SPPE.src.utils.img import im_to_torch
from queue import Queue, LifoQueue
from pPose_nms import write_json
from fn import getTime

# #############################
from src.init import init_frames  # 初始化器
from src.tools import load_data  # 正常帧加载器
from src.tools import Visualize  # 可视化过程，其中含有路径配置
from src.match import Hungarian_match, Feature_match, Update_tracker  # 单通道匹配函数
from src.match import Inter_cam_match_1, Inter_cam_match_2
from src.tools import YOLO
import time

cam_number = 1
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

class for_store_match:
    def __init__(self):
        self.Q = LifoQueue(maxsize=1024)      # for update match_id_cam,match_id
    def write(self,match_id_cam,match_id):
        self.Q.put((match_id_cam,match_id))
    def read(self):
        return self.Q.get()
    def isfull(self):
        return self.Q.full()
class for_store_tracker:
    def __init__(self):
        self.Q = LifoQueue(maxsize=1024)      # for update match_id_cam,match_id
    def write(self,Frames_Lib,tracker,tracker_id,tracker_cnt):
        self.Q.put((Frames_Lib,tracker,tracker_id,tracker_cnt))
    def read(self):
        return self.Q.get()
    def isfull(self):
        return self.Q.full()

if __name__ == '__main__':  

    url_1 = "rtsp://linye:linye123@192.168.200.253:554/Streaming/Channels/101"
    # url_1 = 0
    url_2 = "rtsp://linye:linye123@192.168.200.253:554/Streaming/Channels/301"
    # url_2 = 0
    webcam = args.webcam
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load input video
    fvs_0 = WebcamLoader(url_1).start()
    fvs_1 = WebcamLoader(url_2).start()

    (fourcc, fps1, frameSize1) = fvs_0.videoinfo()
    (fourcc, fps2, frameSize2) = fvs_1.videoinfo()
    # read the camera parameter of this dataset
    # with open ( opt.camera_parameter_path,'rb' ) as f:
    #     camera_parameter = pickle.load (f)

    # detection module
    print('Loading detection model ')   
    sys.stdout.flush()
    det_loader_1 = DetectionLoader(fvs_0, batchSize=1).start()
    det_loader_2 = DetectionLoader(fvs_1, batchSize=1).start()

    save_path = os.path.join(args.outputpath, 'AlphaPose_webcam'+webcam+'.avi')
    # writer1 = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps1, frameSize1).start()
    # writer2 = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps2, frameSize2).start()

    runtime_profile = {
        'ld': [],
        'dt': [],
        'dn': [],
        'pt': [],
        'pn': []
    }
    def loop():
        n = 0
        while True:
            yield n
            n += 1
    print('Initing tracker...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
    store_tracker1 = for_store_tracker()
    store_tracker2 = for_store_tracker()
    for i in im_names_desc:
        try:
            with torch.no_grad():
                (orig_img,frame_id,bbox,bbox_score,kp,kp_score,imgs) = det_loader_1.read()
                if bbox == None or len(bbox) == 0:
                    continue
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
                        continue
                # 处于正常处理状态
                elif init_flag == 0:
                    iouloader1 = IOULoader(store_tracker1,det_loader_1,Frames_Lib,tracker,tracker_id, tracker_cnt, feature_model)
                    store_tracker1.write(Frames_Lib,tracker,tracker_id,tracker_cnt)
                    print('init success')
                    break
        except KeyboardInterrupt:
            exit('Initing fail...')
    cam_number = 1
    match_id_cam, match_id = [], []  # 记录五帧内帧间关联信息（由mvpose结果进行联合分析）
    init_flag = 1  # 是否在初始化追踪器期间，默认处于
    init_info = [[], [], [], []]  # 对初始化25帧进行记录
    im_names_desc = tqdm(loop())
    for i in im_names_desc:
        try:
            with torch.no_grad():
                (orig_img,frame_id,bbox,bbox_score,kp,kp_score,imgs) = det_loader_2.read()
                if bbox == None or len(bbox) == 0:
                    continue
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
                        continue
                # 处于正常处理状态
                elif init_flag == 0:
                    iouloader2 = IOULoader(store_tracker2,det_loader_2,Frames_Lib,tracker,tracker_id, tracker_cnt, feature_model)
                    store_tracker2.write(Frames_Lib,tracker,tracker_id,tracker_cnt)
                    print('init success')
                    break
        except KeyboardInterrupt:
            exit('Initing fail...')
    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
    iouloader1.start();iouloader2.start()
    store_match1 = for_store_match()
    store_match1.write([],[])
    store_match2 = for_store_match()
    store_match2.write([],[])
    featureloader1 = FeatureLoader(store_tracker1,iouloader1,store_match1).start()
    featureloader2 = FeatureLoader(store_tracker2,iouloader2,store_match2).start()
    for i in im_names_desc:
        try:
            start_time = time.time()
            with torch.no_grad():
                
                #orig_img,box_1,box_s_1,roi_1,kp_1,kp_s_1 = pose_loader_1.read()
                #(result, orig_img, im_name) = det_loader_1.read()
                # writer1.save(result, orig_img, str(i)+'.jpg')
                # (result, orig_img, im_name) = det_loader_2.read()
                # writer2.save(result, orig_img, str(i)+'.jpg')
                # (orig_img,frame_id,bbox,bbox_score,kp,kp_score,imgs) = det_loader_1.read()
                (orig_img1,frame_id1,cam_number,ID_list_cam1, match_id_cam1,match_id1,\
                    features1,tracker1,tracker_id1,tracker_cnt1) = featureloader1.read()
                (orig_img2,frame_id2,cam_number,ID_list_cam2, match_id_cam2,match_id2,\
                        features2,tracker2,tracker_id2,tracker_cnt2) = featureloader2.read()
                # 多通道联合分析（首先生成ID链接信息，后生成总的ID）
                # print('ID_list_cam1:')  # [[1]]
                # print(tracker1)         # ROI坐标

                tracker = [tracker1[0],tracker2[0]]
                ID_list_cam = [ID_list_cam1[0],ID_list_cam2[0]]
                features = [features1[0],features2[0]]
                match_id_cam1.extend(match_id_cam2)
                match_id1.extend(match_id2)
                tracker_id = [tracker_id1[0],tracker_id2[0]]
                match_id_cam = match_id_cam1
                match_id = match_id1
                # if tracker_id == None:
                #     continue
                
                match_id_cam, match_id = Inter_cam_match_1(cam_number, frame_id, ID_list_cam, \
                    match_id_cam,match_id,features)  # 获取
            
                fuse_ID = Inter_cam_match_2(tracker, tracker_id, match_id_cam, match_id)
                
                #Visualize([orig_img1,orig_img2],2, frame_id, tracker, tracker_id, fuse_ID)
            end_time = time.time()
            print('   %f FPS' % (1 / (end_time - start_time)))

        except KeyboardInterrupt:
            break
