from PIL import Image
import json
from Mvpose.get_affinity_mat import get_reid_affinity
from Mvpose.get_match_list import get_match_list
from DGnet.reid_interface import ReID  # ReID模型
import time


class Frames():
    def __init__(self, cam_number, dimGroup_init, rois_init, features_init, imgs_init, match_list_cam):
        """根据前25帧信息初始化"""
        self.cam_number = cam_number  # 记录摄像机数目
        self.rois = [[] for i in range(self.cam_number)]  # 25帧内的RoI
        self.features = [[] for i in range(self.cam_number)]  # 25帧内的feature
        self.ids = [[] for i in range(self.cam_number)]  # 25帧内ID
        self.imgs = [[] for i in range(self.cam_number)]  # 存放RoI图片

        self.match_list_cam = match_list_cam

        for cam in range(self.cam_number):
            ID = [-1 for i in range(0, dimGroup_init[cam][-1])]  # 5帧所有RoI全部置为-1
            for person_id, person_roi in enumerate(match_list_cam[cam]):  # 以人为对象进行考察
                for i in range(0, dimGroup_init[cam][-1]):
                    if i in person_roi.tolist():
                        ID[i] = person_id  # roi在匹配列表中，返回此人的ID值

            for i in range(0, len(dimGroup_init[cam]) - 1):
                roi_, features_, ids_, imgs_ = [], [], [], []
                for roi_id in range(dimGroup_init[cam][i], dimGroup_init[cam][i + 1]):
                    roi_.append(rois_init[cam][roi_id])
                    features_.append(features_init[cam][roi_id])
                    ids_.append(ID[roi_id])
                    imgs_.append(imgs_init[cam][roi_id])
                self.rois[cam].append(roi_)
                self.features[cam].append(features_)
                self.ids[cam].append(ids_)
                self.imgs[cam].append(imgs_)

    def get_init_tracker(self, tracker, tracker_id, tracker_cnt):
        """生成追踪器"""
        for cam in range(self.cam_number):
            tracker[cam] = self.rois[cam][-1]
            tracker_id[cam] = self.ids[cam][-1]
            tracker_cnt[cam] = len(self.match_list_cam[cam])
        return tracker, tracker_id, tracker_cnt

    def get_frames(self):
        """获取五帧信息"""
        rois_5_cam, features_5_cam, ids_5_cam = [], [], []
        dimGroup_5_cam = [[0] for i in range(self.cam_number)]
        for cam in range(self.cam_number):
            rois_5, features_5, ids_5 = [], [], []
            for frame in range(0, 25, 5):  # 5帧放入

                for i in range(len(self.rois[cam][frame])):  # 一帧内含有的roi
                    rois_5.append(self.rois[cam][frame][i])
                    features_5.append(self.features[cam][frame][i])  # 此处需要卷积操作
                    ids_5.append(self.ids[cam][frame][i])
                dimGroup_5_cam[cam].append(dimGroup_5_cam[cam][-1] + len(self.rois[cam][frame]))  # roi累加计数
            rois_5_cam.append(rois_5)
            features_5_cam.append(features_5)
            ids_5_cam.append(ids_5)
        return rois_5_cam, features_5_cam, ids_5_cam, dimGroup_5_cam

    def update_current_frame(self, ID_list_cam, rois, features, tracker, tracker_id):
        """根据ID列表进行Frame类更新"""
        for cam in range(self.cam_number):
            # 末尾追加当前帧信息
            self.rois[cam].append(rois[cam])
            self.features[cam].append(features[cam])
            self.ids[cam].append(ID_list_cam[cam])

            # 删除首帧信息
            del (self.rois[cam][0])
            del (self.features[cam][0])
            del (self.ids[cam][0])

            # 更新tracker信息
            tracker[cam] = self.rois[cam][-1]
            tracker_id[cam] = self.ids[cam][-1]
        return tracker, tracker_id


def init_frames(cam_number, init_info):
    """首25帧初始化，为了防止首25帧无RoI，等待到每个视角都有3个人以上时启动初始化算法"""

    # 生成单通道独立追踪列表
    tracker = [[] for i in range(cam_number)]  # 追踪器
    tracker_id = [[] for i in range(cam_number)]  # 追踪器指派的ID
    tracker_cnt = [0 for i in range(cam_number)]  # 摄像头中人数

    rois_init, features_init, dimGroup_init = [[], [], []], [[], [], []], [[0], [0], [0]]

    # 创建Re-ID模型
    feature_model = ReID()  # 载入Re-ID模型

    def roi_th_reid(cam, frame_id, init_info, th=0.3):  # 单帧数据
        frame_id, bbox, bbox_score, img = init_info[0][frame_id], init_info[1][frame_id][cam], \
                                            init_info[2][frame_id][cam], init_info[3][frame_id][cam]
        if bbox == None or len(bbox) == 0 :
            return [], [], [], []
        else:
            rois, id, feature, imgs = [], [], [], []
            for roi_id in range(len(bbox)):
                if bbox_score[roi_id] > th:  # 有效检测
                    rois.append(bbox[roi_id])
                    id.append(roi_id)
                    # 将imgs读进去
                    imgs.append(img[roi_id])

            # 对imgs进行卷积处理
            start_time = time.time()
            
            try:
                feature = feature_model.get_feature(imgs)
            except:
                print(imgs)
                print(len(imgs))
                
            end_time = time.time()
            print('      using %fs' % (end_time - start_time))

            return rois, id, feature, imgs

    rois_init, features_init, imgs_init, dimGroup_init = \
        [[] for i in range(cam_number)], [[] for i in range(cam_number)], [[] for i in range(cam_number)], \
        [[0] for i in range(cam_number)]  # 初始化

    # 读取每帧信息
    for cam in range(cam_number):
        for frame_id in range(25):  # 遍历25帧，获取ID信息

            roi, id, feature, img = roi_th_reid(cam, frame_id, init_info, th=0.3)

            rois_init[cam].extend(roi)  # 将每一帧中的roi无差别放入列表中
            dimGroup_init[cam].append(dimGroup_init[cam][-1] + len(roi))  # 对维度组进行记录
            features_init[cam].extend(feature)
            imgs_init[cam].extend(img)

    print('**** start_frame:')
    print(len(imgs_init))
    print(len(imgs_init[0]))
 

    if len(features_init[0]) > 3 :  # 若满足初始化条件    # and len(features_init[2]) > 3
        init_flag = 0

        # 使用Mvpose生成匹配结果
        match_list_cam = []
        for cam in range(0, cam_number):
            if len(features_init[cam]) == 0:
                continue
            affinity = get_reid_affinity(features_init[cam])
            match_list = get_match_list(affinity, dimGroup=dimGroup_init[cam])
            match_list_cam.append(match_list)

        # 存储25f信息，roi、feature、id、imgs信息
        Frames_Lib = Frames(cam_number, dimGroup_init, rois_init, features_init, imgs_init, match_list_cam)
        tracker, tracker_id, tracker_cnt = Frames_Lib.get_init_tracker(tracker, tracker_id, tracker_cnt)  # 初始化追踪器

        return init_flag, Frames_Lib, tracker, tracker_id, tracker_cnt, feature_model

    else:  # 不满足初始条件
        init_flag = 1
        return init_flag, [], [], [], [], []
    