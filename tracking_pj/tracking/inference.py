from yolo.darknet import Darknet
from opt import opt
import torch
from torch.autograd import Variable
from yolo.util import write_results, dynamic_write_results
from yolo.preprocess import prep_frame
from pPose_nms import pose_nms
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
from dataloader import crop_from_dets, Mscoco

args = opt
class yolo_detecter:
    def __init__(self):
        det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        det_model.load_weights('models/yolo/yolov3-spp.weights')
        det_model.net_info['height'] = args.inp_dim
        det_inp_dim = int(det_model.net_info['height'])
        assert det_inp_dim % 32 == 0
        assert det_inp_dim > 32
        self.det_inp_dim = det_inp_dim
        det_model.cuda()
        det_model.eval()
        self.det_model = det_model

    def get_model(self):
        return self.det_model
    def get_prediction(self,img, cuda=True):
        prediction = self.det_model(img, CUDA=cuda)
        return prediction
    def NMS_process(self,prediction,frame_id,fvis_0,fvis_1,im_dim_list):
        dets = dynamic_write_results(prediction, args.confidence, args.num_classes, nms=True, nms_conf=args.nms_thesh)
        if (isinstance(dets, int) or dets.shape[0] == 0):
            return False,None,None

        im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
        scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

        # coordinate transfer
        dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
        dets[:, 1:5] /= scaling_factor
        for j in range(dets.shape[0]):
            dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
            dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
        boxes = dets[:, 1:5].cpu()
        scores = dets[:, 5:6].cpu()
        return True,boxes,scores

class pose_detection:
    def __init__(self):
        pose_dataset = Mscoco()
        if args.fast_inference:  # True
            pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        pose_model.cuda()
        pose_model.eval()
        self.pose_model = pose_model
    def get_model(self):
        return self.pose_model
    def get_prediction(self,orig_img,boxes,scores,single_height,output_l):
        inp = im_to_torch(orig_img)
        inps = torch.zeros(boxes.size(0), 3, args.inputResH, args.inputResW)
        pt1 = torch.zeros(boxes.size(0), 2)
        pt2 = torch.zeros(boxes.size(0), 2)
        inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
        inps = Variable(inps.cuda())

        hm = self.pose_model(inps)
        if boxes is None:
            return False,hm
        else:
            preds_hm, preds_img, preds_scores = getPrediction(hm.cpu(), pt1, pt2, args.inputResH, args.inputResW, args.outputResH, args.outputResW)
            # result(被姿态估计筛选后的) : bbox,bbox_score,roi,keypoints, kp_score (两个镜头的)
            box, box_s, roi, kp, kp_s = pose_nms(boxes, scores, preds_img, preds_scores, single_height, orig_img,output_l)
            return True,(box, box_s, roi, kp, kp_s)
