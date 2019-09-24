import cv2
import json
import numpy as np
import sys
import time
import torch.multiprocessing as mp
from SPPE.src.main_fast_inference import *
from yolo.util import write_results, dynamic_write_results
from pPose_nms import pose_nms
import torchvision.transforms as transforms
from SPPE.src.utils.img import im_to_torch

from pPose_nms import write_json
from fn import getTime
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread
import os
import torch
from torch.autograd import Variable
import torch.utils.data as data

from SPPE.src.utils.img import cropBox, im_to_torch
from opt import opt
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue
if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame
from yolo.darknet import Darknet
# from reid import reid_interface
from yolo.preprocess import prep_frame
# #############################
from src.init import init_frames  # 初始化器
from src.tools import load_data  # 正常帧加载器
from src.tools import Visualize  # 可视化过程，其中含有路径配置
from src.match import Hungarian_match, Feature_match, Update_tracker  # 单通道匹配函数
from src.match import Inter_cam_match_1, Inter_cam_match_2
from src.tools import YOLO
import time
from threading import Thread,Lock

class WebcamLoader:
    def __init__(self, webcam, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(webcam)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,384)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,218)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.batchSize = 1
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        i = 0
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.2), int(frame.shape[0] * 0.2)), interpolation = cv2.INTER_AREA)
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                inp_dim = int(opt.inp_dim)
                img, orig_img, dim = prep_frame(frame, inp_dim)

                im_name = str(i)+'.jpg' 
                
                with torch.no_grad():
                    # Human Detection
                    
                    im_dim_list = torch.FloatTensor([dim]).repeat(1, 2)
                    self.Q.put((img, orig_img, im_name, im_dim_list))
                    i = i+1
            else:
                with self.Q.mutex:
                    self.Q.queue.clear()
    def videoinfo(self):
        # indicate the video info
        fourcc=int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps=self.stream.get(cv2.CAP_PROP_FPS)
        frameSize=(int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc,fps,frameSize)

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue size
        return self.Q.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class DetectionLoader:
    def __init__(self, dataloder, batchSize=1, queueSize=1024):
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()
        self.stopped = False
        self.dataloder = dataloder
        self.batchSize = batchSize
        # initialize the queue used to store frames read from
        # the video file
        self.Q = LifoQueue(maxsize=queueSize)
        pose_dataset = Mscoco()
        if opt.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            (img, orig_img, im_name, im_dim_list) = self.dataloder.getitem()
             
            with self.dataloder.Q.mutex:
                self.dataloder.Q.queue.clear()
            with torch.no_grad():
                # Human Detection
                #img = img.cuda()
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
                # im_dim_list = im_dim_list.cuda()
                frame_id = int(im_name.split('.')[0])
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                             opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img,frame_id,None,None,None,None,None))
                    continue

                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                
                boxes = dets[:, 1:5] 
                scores = dets[:, 5:6] 
                # Pose Estimation
                inp = im_to_torch(orig_img)
                inps = torch.zeros(boxes.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes.size(0), 2)
                pt2 = torch.zeros(boxes.size(0), 2)
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
                inps = Variable(inps.cuda())
                hm = self.pose_model(inps)
                if boxes is None:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img,frame_id,None,None,None,None,None))
                    continue
                else:
                    preds_hm, preds_img, preds_scores = getPrediction(hm.cpu(), pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                    bbox,b_score,kp,kp_score,roi = pose_nms(orig_img,boxes, scores, preds_img, preds_scores)
                    # result = {
                    #     'imgname': im_name,
                    #     'result': result,
                    #     'orig_img' : orig_img
                    # }
                     
                if self.Q.full():
                    time.sleep(2)
                #self.Q.put((orig_img[k], im_name[k], boxes_k, scores[dets[:,0]==k], inps, pt1, pt2))
                #self.Q.put((result, orig_img, im_name))
                self.Q.put((orig_img,frame_id,bbox,b_score,kp,kp_score,roi))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()

class IOULoader:
    def __init__(self, store_tracker,detectionLoader,Frames_lib,tracker,tracker_id, tracker_cnt, feature_model, queueSize=1024):
        self.detectionLoader = detectionLoader
        self.stopped = False
        self.Q = LifoQueue(maxsize=queueSize)
        self.Frames_Lib = Frames_lib
        self.store_tracker = store_tracker
        self.feature_model = feature_model
        self.cam_number = 1

    def start(self):
        # start a thread to read frames from the file video stream
        
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            (orig_img,frame_id,bbox,bbox_score,kp,kp_score,imgs) = self.detectionLoader.read()
            (Frames_Lib,tracker,tracker_id,tracker_cnt) = self.store_tracker.read()
            #print('iou tracker_id')
            with self.detectionLoader.Q.mutex:
                self.detectionLoader.Q.queue.clear()
            with torch.no_grad():
                if bbox == None or len(bbox)==0:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img,frame_id,self.cam_number,Frames_Lib,[[]],[[]],[[]],tracker,tracker_id, tracker_cnt))
                    continue
                rois, ids, features, imgs,score_cam = load_data(self.cam_number, frame_id, bbox, bbox_score, imgs,\
                    self.feature_model)  # 获取当前时刻roi信息
                # 单通道IOU追踪更新
                ID_list_cam = Hungarian_match(rois, self.cam_number, tracker,tracker_id)  # 根据匈牙利匹配算法对IoU进行匹配，得到ID列表，不确定的为-1
                if self.Q.full():
                    time.sleep(2)    
                self.Q.put((orig_img,frame_id,self.cam_number, Frames_Lib, ID_list_cam, rois, features,\
                        tracker,tracker_id, tracker_cnt))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()            

class FeatureLoader:
    def __init__(self,store_tracker, iouLoader,store_match, queueSize=1024):
        self.iouLoader = iouLoader
        self.stopped = False
        self.Q = LifoQueue(maxsize=queueSize)
        self.cam_number = 1
        self.frame_inter = 3
        self.store_match = store_match
        self.store_tracker = store_tracker

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):      
        count = 0
        while True:
            count += 1
            if count  :
                match_id_cam,match_id = self.store_match.read()
                (orig_img,frame_id,cam_number, Frames_Lib, ID_list_cam, rois, features,\
                            tracker,tracker_id,tracker_cnt) = self.iouLoader.read()
                # print(ID_list_cam) # [[1]] 表示当前镜头下的人的ID，如ID为1
                with self.iouLoader.Q.mutex:
                    self.iouLoader.Q.queue.clear()
                with torch.no_grad():
                    ID_list_cam = Feature_match(cam_number, Frames_Lib, ID_list_cam, rois, features)  # 使用25f法对IoU方法进行修正
                    # 更新Frame(时序更新)
                    # 检测ID_list_cam中是否有-1，有的话新增ID，重新更新ID列表
                    # 将当前帧添加到Frame_lib中，删除最早的帧，保持lib中有25帧
                    Update_tracker(cam_number, Frames_Lib, ID_list_cam, rois, features, tracker, tracker_id,
                                    tracker_cnt)   
                    if self.Q.full():
                        time.sleep(2)    
                    self.Q.put((orig_img,frame_id,cam_number,ID_list_cam, match_id_cam,match_id,\
                        features,tracker,tracker_id,tracker_cnt))
                    if self.store_match.isfull():
                        time.sleep(2)
                    self.store_match.write(match_id_cam,match_id)
                    if self.store_tracker.isfull():
                        time.sleep(2)
                    self.store_tracker.write(Frames_Lib,tracker,tracker_id,tracker_cnt)
    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()

class PoseLoader:
    def __init__(self, detectionLoader, single_height,queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.detectionLoader = detectionLoader
        self.stopped = False
        self.batchSize = opt.posebatch
        self.single_height = single_height
        # initialize the queue used to store data
        self.Q = LifoQueue(maxsize=queueSize)
        # Load pose model
        pose_dataset = Mscoco()
        if opt.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping the whole dataset
        while True:
            print('read det')
            (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()
            print('read det over ')
            with self.detectionLoader.Q.mutex:
                self.detectionLoader.Q.queue.clear()
            with torch.no_grad():
                
                if boxes is None or boxes.nelement() == 0:
                    #writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    # self.Q.put((orig_img,None,None,None,None,None))
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img,None,None,None,None,None))
                    continue
                
                datalen = inps.size(0)  # batch数据size
                leftover = 0
                if (datalen) % self.batchSize:
                    leftover = 1
                num_batches = datalen // self.batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j*self.batchSize:min((j +  1)*self.batchSize, datalen)].cuda()
                    hm_j = self.pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                hm = hm.cpu().data
     
                preds_hm, preds_img, preds_scores = getPrediction(hm, pt1, pt2, opt.inputResH,
                                                                      opt.inputResW, opt.outputResH, opt.outputResW)
                # result(被姿态估计筛选后的) : bbox,bbox_score,roi,keypoints, kp_score (两个镜头的)
                box,box_s,roi,kp,kp_s = pose_nms(boxes.cpu(), scores.cpu(), preds_img.cpu(), preds_scores.cpu()\
                    ,self.single_height,orig_img)
                if self.Q.full():
                    time.sleep(2)
                self.Q.put((orig_img,box,box_s,roi,kp,kp_s))
                # if boxes is None or boxes.nelement() == 0:
                #     while self.Q.full():
                #         time.sleep(0.2)
                #     self.Q.put((None, orig_img, im_name, boxes, scores, None, None))
                #     continue
                # inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                # inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                # while self.Q.full():
                #     time.sleep(0.2)
                # self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DataWriter:
    def __init__(self, save_video=False,
                savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480),
                queueSize=1024):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                
                (result, orig_img, im_name) = self.Q.get()

                if result is None:
                    if opt.save_img or opt.save_video or opt.vis:
                        img = orig_img
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
                else:
                    import pickle
                    with open("result.pkl", "wb") as f:
                        pickle.dump(result, f)

                    self.final_result.append(result)
                    if opt.save_img or opt.save_video or opt.vis:
                        img = vis_frame(orig_img, result)
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
            else:
                time.sleep(0.05)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    # def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
    #     # save next frame in the queue
    #     self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def save(self, result, orig_img, im_name):
        # save next frame in the queue
        self.Q.put((result, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()
def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''
    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2

class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
