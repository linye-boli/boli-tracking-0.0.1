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
from dataloader import WebcamLoader, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
# from reid import reid_interface
from yolo.preprocess import prep_frame
from SPPE.src.main_fast_inference import *
from yolo.util import write_results, dynamic_write_results
from pPose_nms import pose_nms
from SPPE.src.utils.img import im_to_torch

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

    (fourcc, fps, frameSize) = fvs_0.videoinfo()

    # read the camera parameter of this dataset
    # with open ( opt.camera_parameter_path,'rb' ) as f:
    #     camera_parameter = pickle.load (f)

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_webcam' + webcam + '.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    # detection module
    print('Loading detection model ')
    sys.stdout.flush()
    det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights('models/yolo/yolov3-spp.weights')
    det_model.net_info['height'] = args.inp_dim
    det_inp_dim = int(det_model.net_info['height'])
    assert det_inp_dim % 32 == 0
    assert det_inp_dim > 32
    det_model.cuda()
    det_model.eval()

    # pose module
    print('Loading pose model')
    sys.stdout.flush()
    pose_dataset = Mscoco()
    if args.fast_inference:  # True
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    # reid module
    # reid_model = reid_interface.ReID(is_folder=False)

    # Running time of each module
    runtime_profile = {
        'ld': [],
        'dt': [],
        'dn': [],
        'pt': [],
        'pn': []
    }

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
    for i in im_names_desc:
        try:
            begin = time.time()
            start_time = getTime()
            frame_0 = fvs_0.read()
            frame_1 = fvs_1.read()
            single_height = frame_0.shape[0]
            print(frame_0.shape) # (432, 768, 3)

            # pre-process
            frame = np.concatenate([frame_0, frame_1], 0)
            inp_dim = int(args.inp_dim)  # default=608
            img, orig_img, dim = prep_frame(frame, inp_dim)
            #print('img:',img.shape)  # torch.Size([1, 3, 608, 608])
            # print('orig_img:',orig_img.shape)  # (864, 768, 3)
            # print('dim',dim)    # (768, 864)

            inp = im_to_torch(orig_img)
            im_dim_list = torch.FloatTensor([dim]).repeat(1, 2)
            # print(im_dim_list) # tensor([[768., 864., 768., 864.]])

            ckpt_time, load_time = getTime(start_time)
            runtime_profile['ld'].append(load_time)
            with torch.no_grad():
                # human detection
                img = Variable(img).cuda()
                im_dim_list = im_dim_list.cuda()


                # ###################
                yolo_start = time.time()
                prediction = det_model(img, CUDA=True)
                yolo_end = time.time()
                _yolo_delta = yolo_end - yolo_start
                #print('######## YOLO_time FPS: %f:'%(1/_yolo_delta))
                # ###################


                ckpt_time, det_time = getTime(ckpt_time)
                runtime_profile['dt'].append(det_time)
                # NMS process
                dets = dynamic_write_results(prediction, args.confidence,
                                             args.num_classes, nms=True, nms_conf=args.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    writer.save(None,  orig_img =orig_img, im_name=str(i) + '.jpg')
                    continue
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5].cpu()
                scores = dets[:, 5:6].cpu()
                # print('boxes:',boxes.shape)
                # print(type(boxes))
                # print(boxes[0])
                # print('---------')
                # print(scores)
                # print(type(scores))

                # Separate frame (目前默认已知两个摄像头)
                boxes_1 = []
                scores_1 = []
                boxes_2 = []
                scores_2 = []
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j]
                    if y1 < single_height and y2 <= single_height:
                        boxes_1.append(boxes[j])
                        scores_1.append(scores[j])
                    elif y1 < single_height and y2 > single_height:
                        boxes_1.append(boxes[j])
                        scores_1.append(scores[j])
                        boxes_1[-1][3] = single_height  # 校正
                    else:
                        boxes_2.append(boxes[j])
                        scores_2.append(scores[j])
                        boxes_2[-1][3] -= single_height
                        boxes_2[-1][1] -= single_height

                ckpt_time, detNMS_time = getTime(ckpt_time)
                runtime_profile['dn'].append(detNMS_time)
                # roi
                roi_1 = []
                roi_2 = []
                orig_img_1 = orig_img[:single_height, :, :]
                orig_img_2 = orig_img[single_height:, :, :]

                for bs_i in range(len(boxes_1)):
                    sn = scores_1[bs_i].numpy()
                    # if(sn > 0.85):  
                    bn = boxes_1[bs_i].numpy()
                    roi_temp = orig_img_1[int(bn[1]):int(bn[3]), int(bn[0]):int(bn[2])]
                    roi_1.append(roi_temp)
                for bs_i in range(len(boxes_2)):
                    sn = scores_2[bs_i].numpy()
                    # if(sn > 0.85):  
                    bn = boxes_2[bs_i].numpy()
                    roi_temp = orig_img_2[int(bn[1]):int(bn[3]), int(bn[0]):int(bn[2])]
                    roi_2.append(roi_temp)
                # pose estimation
                inps = torch.zeros(boxes.size(0), 3, args.inputResH, args.inputResW)
                pt1 = torch.zeros(boxes.size(0), 2)
                pt2 = torch.zeros(boxes.size(0), 2)
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
                inps = Variable(inps.cuda())

                hm = pose_model(inps)
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)

                if boxes is None:
                    result = None
                else:
                    # Get keypoint location from heatmaps
                    preds_hm, preds_img, preds_scores = getPrediction(hm.cpu(), pt1, pt2, args.inputResH,
                                                                      args.inputResW, args.outputResH, args.outputResW)
                    # result : keypoints, kp_score, proposal_score
                    result = pose_nms(boxes, scores, preds_img, preds_scores)
                    # print('result:',result) # result: [{'keypoints':...每一个ROI为一个字典，包含kp等数据
                    # print('-------------')
                    # print('result:',result[0]) # 第一个ROI
                    
                im_name = str(i) + '.jpg'
                # result = {
                    
                #     'result': result,
                #     'orig_img': orig_img,
                #     'imgname': im_name
                # }
                # writer.save(boxes, scores, hm.cpu(), pt1, pt2, orig_img, im_name=str(i)+'.jpg')
                # writer.save(result, orig_img, str(i) + '.jpg')
                writer.save(result,orig_img=orig_img,im_name=im_name)
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)
                # reid 
                data_resize = transforms.Compose([
                    transforms.Resize((256, 128), interpolation=3),
                ])
                roi_batch1 = []
                for j in range(len(roi_1)):
                    #print('ia m here')
                    temp_roi = Image.fromarray(roi_1[j].astype('uint8')).convert('RGB')
                    temp_roi = data_resize(temp_roi)
                    roi_batch1.append(temp_roi)
                roi_batch2 = []
                for j in range(len(roi_2)):
                    #print('ia m here')
                    temp_roi = Image.fromarray(roi_2[j].astype('uint8')).convert('RGB')
                    temp_roi = data_resize(temp_roi)
                    roi_batch2.append(temp_roi)    
                # reid_feature = reid_model.get_feature(roi_batch1)
                # print('iam over')
                # print(reid_feature)
                # print(reid_feature.shape)
                # print(type(reid_feature)) 

                # matching
                for_matching_input = {
                    'cam_number': num_cam,
                    'frame_id': i,
                    'bbox': [boxes_1, boxes_2],
                    'bbox_score': [scores_1, scores_2],
                    'imgs': [roi_batch1, roi_batch2]
                }
                frame_id, bbox, bbox_score, imgs = i, [boxes_1, boxes_2], [scores_1, scores_2], [roi_batch1, roi_batch2]

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
                        init_flag, Frames_Lib, tracker, tracker_id, tracker_cnt, feature_model = init_frames(
                            cam_number, init_info)
                        init_info = [[], [], [], []]  # 记录器清零
                        continue

                # # 处于正常处理状态
                elif init_flag == 0:
                    # ###################
                    load_start = time.time()
                    rois, ids, features, imgs, score_cam = load_data(cam_number, frame_id, bbox, bbox_score, imgs,
                                                                     feature_model)  # 获取当前时刻roi信息
                    Load_end = time.time()
                    Load_delta = Load_end - load_start
                    print('Load_time : %f' % (1 / Load_delta))
                    # ###################

                    # ###################
                    load_start = time.time()
                    # 单通道追踪更新
                    ID_list_cam = Hungarian_match(rois, cam_number, tracker,
                                                  tracker_id)  # 根据匈牙利匹配算法对IoU进行匹配，得到ID列表，不确定的为-1
                    Load_end = time.time()
                    Load_delta = Load_end - load_start
                    print('Hungarian_time : %f' % (1 / Load_delta))
                    # ###################

                    # ###################
                    load_start = time.time()
                    ID_list_cam = Feature_match(cam_number, Frames_Lib, ID_list_cam, rois, features)  # 使用25f法对IoU方法进行修正
                    Load_end = time.time()
                    Load_delta = Load_end - load_start
                    print('Match_time : %f' % (1 / Load_delta))
                    # ###################

                    # ###################
                    load_start = time.time()
                    # 更新Frame(时序更新)
                    Update_tracker(cam_number, Frames_Lib, ID_list_cam, rois, features, tracker, tracker_id,
                                   tracker_cnt)
                    Load_end = time.time()
                    Load_delta = Load_end - load_start
                    print('Update_time : %f' % (1 / Load_delta))
                    # ###################

                    # 多通道联合分析（首先生成ID链接信息，后生成总的ID）
                    match_id_cam, match_id = Inter_cam_match_1(cam_number, frame_id, ID_list_cam, match_id_cam, match_id, features)  # 获取
                    fuse_ID = Inter_cam_match_2(tracker, tracker_id, match_id_cam, match_id)

                    # 进行可视化
                    for_visual_input = {
                        'bbox': tracker,  # [cam][bbox]
                        'bbox_id': tracker_id,  # [cam][id]
                        'bbox_score': score_cam,  # [cam][score]
                        'ori_img': [frame_0, frame_1]

                    }
                    # ###################
                    load_start = time.time()
                    Visualize([frame_0, frame_1], cam_number, frame_id, tracker, tracker_id, fuse_ID)
                    Load_end = time.time()
                    Load_delta = Load_end - load_start
                    print('Visualize_time : %f' % (1 / Load_delta))
                    # ###################

            delta = end - begin
            print('######## ALL_time: FPS: %f:'%(1/delta))

            # TQDM
            # im_names_desc.set_description(
            #     'load time: {ld:.4f} | det time: {dt:.4f} | det NMS: {dn:.4f} | pose time: {pt:.4f} | post process: {pn:.4f}'.format(
            #         ld=np.mean(runtime_profile['ld']), dt=np.mean(runtime_profile['dt']), dn=np.mean(runtime_profile['dn']),
            #         pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            # )
        except KeyboardInterrupt:
            break
