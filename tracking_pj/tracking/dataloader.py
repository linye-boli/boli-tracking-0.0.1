import cv2
import json
import numpy as np
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread
import os
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from SPPE.src.utils.img import cropBox, im_to_torch
from pPose_nms import pose_nms, write_json
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


class WebcamLoader:
    def __init__(self, webcam, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(webcam)

        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,384)       
                # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,218)
                # print('framesize:')
                # print(int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))
                # input()

                (grabbed, frame) = self.stream.read()
                # frame = cv2.resize(frame, (int(frame.shape[1] * 0.25), int(frame.shape[0] * 0.25)), interpolation=cv2.INTER_AREA)

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                self.Q.put(frame)
            else:
                with self.Q.mutex:
                    self.Q.queue.clear()

    def videoinfo(self):
        # indicate the video info
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frameSize = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc, fps, frameSize)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue size
        return self.Q.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class DataWriter:
    def __init__(self, save_video=False,
                 savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640, 480),
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
        self.img_folder = '../data/coco/images'  # root image folders
        self.is_train = train  # training set or test set
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
