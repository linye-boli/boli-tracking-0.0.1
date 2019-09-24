from Mvpose.get_affinity_mat import get_reid_affinity, geometry_affinity
from Mvpose.get_match_list import get_match_list
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy


# 单通道匹配更新
def Hungarian_match(rois, cam_number, tracker, tracker_id):
    """使用匈牙利算法进行相邻两帧匹配，后检查匹配列表是否合法"""

    def compute_iou(rec1, rec2):
        """计算交叠区域占rec1的比例，再取反，最后交叠越大值越小，为Hungarian算法服务"""
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2
        # sum_area = S_rec1

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 1
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return 1.001 - intersect / (sum_area - intersect)

    # 生成RoI交叠信息矩阵
    IoU_mat_cam = []
    for cam in range(cam_number):
        IoU = []
        for i in range(len(tracker[cam])):
            _ = []
            for j in range(len(rois[cam])):
                _.append(compute_iou(tracker[cam][i], rois[cam][j]))
            IoU.append(_)
        IoU_mat_cam.append(IoU)  # 结果矩阵以行为索引

    # 根据RoI交叠矩阵进行匹配
    ID_list_cam = []
    for cam in range(cam_number):
        ID_list = [-1 for i in range(len(rois[cam]))]  # ID指派列表，当前帧的ID分配情况

        # 记录Tracker和roi匹配情况
        if len(IoU_mat_cam[cam]) != 0:
            row, col = linear_sum_assignment(IoU_mat_cam[cam])  # 使用匈牙利匹配算法完成匹配
            mis_match = []  # 记录错误匹配
            for i in range(len(row)):
                if IoU_mat_cam[cam][row[i]][col[i]] < 0.9:  # IoU明显大于0.1，正确匹配
                    ID_list[col[i]] = tracker_id[cam][row[i]]
        ID_list_cam.append(ID_list)

    return ID_list_cam


def Feature_match(cam_number, Frames_Lib, ID_list_cam, rois, features):
    """使用25f法对IoU方法进行修正"""
    rois_5_cam, features_5_cam, ids_5_cam, dimGroup_5_cam = Frames_Lib.get_frames()  # 获取五帧信息

    roi_feature_list_cam = []  # 需要进行特征匹配的roi索引
    for cam in range(0, cam_number):
        roi_feature_list = []
        for i in range(len(ID_list_cam[cam])):
            if ID_list_cam[cam][i] == -1:
                roi_feature_list.append(i)
        roi_feature_list_cam.append(roi_feature_list)

    features_5_expend, ids_5_expend = [], []  # 将特征、id展开，为Re-ID匹配做准备
    for cam in range(cam_number):
        for i in range(len(features_5_cam[cam])):
            features_5_expend.append(features_5_cam[cam][i])
            ids_5_expend.append(ids_5_cam[cam][i])

    for cam in range(0, cam_number):  # 将每个未知RoI与5帧信息进行mvpose分析
        for i in roi_feature_list_cam[cam]:  # 获取cam视角下当前帧的不确定roi索引

            # ####################### 特征时序匹配 #######################
            distance_list = []
            for id in range(len(features_5_expend)):
                distance_list.append(10 * np.sqrt(np.sum(np.square(np.array(features_5_expend[id]) - np.array(features[cam][i])))))

            if len(distance_list) > 0:  # 若历史25帧中有人才会匹配，没人直接跳过
                if min(distance_list) < 20:  # 距离小于20认定为同一个人
                    id_index = distance_list.index(min(distance_list))
                    if ids_5_expend[id_index] not in ID_list_cam[cam]:
                        ID_list_cam[cam][i] = ids_5_expend[id_index]

            # # ####################### Mvpose时序匹配 #######################
            # features_5_cam[cam].append(features[cam][i])  # 将当前帧的信息放在末尾
            # dimGroup_5_cam[cam].append(dimGroup_5_cam[cam][-1] + 1)  # 维度信息更新
            # # 运行匹配算法
            # affinity = get_reid_affinity(features_5_cam[cam])
            # match_list = get_match_list(affinity, dimGroup=dimGroup_5_cam[cam])
            # # 还原存储信息
            # del (features_5_cam[cam][-1])
            # del (dimGroup_5_cam[cam][-1])
            # # 根据匹配结果指定ID
            # for person in match_list:
            #     if (dimGroup_5_cam[cam][-1] in person) and (len(person) > 1):  # 完成匹配
            #         for per in person:
            #
            #             if per != dimGroup_5_cam[cam][-1]:  # 找到同ID的roi
            #                 if ids_5_cam[cam][per] not in ID_list_cam[cam]:  # 如果没发生ID重合情况
            #                     ID_list_cam[cam][i] = ids_5_cam[cam][per]  # 根据roi索引出ID，赋给当前帧roi

    return ID_list_cam


def Update_tracker(cam_number, Frames_Lib, ID_list_cam, rois, features, tracker, tracker_id, ID_max):
    # 检测ID_list_cam中是否有-1，有的话新增ID，重新更新ID列表
    for cam in range(0, cam_number):
        for i in range(len(ID_list_cam[cam])):
            if ID_list_cam[cam][i] == -1:  # 检测到新建项，赋予ID
                ID_list_cam[cam][i] = Frames_Lib.get_all_IDs(ID_max)[0]

    # 更新Frame类和tracker
    Frames_Lib.update_current_frame(ID_list_cam, rois, features, tracker, tracker_id)


# 多通道融合（每5帧进行一次更新）
def Inter_cam_match_1(cam_number, frame_id, ID_list_cam, match_id_cam, match_id, features, keyp, F):
    """摄像头内独立追踪经过摄像机联合出ID链接信息"""

    # 总人数少于4人且每个cam下人数小于2，不使用Mvpose匹配，改用跨摄像头匹配
    if frame_id % 2 == 0 and (sum([len(i) for i in ID_list_cam]) <= 4) and (len(ID_list_cam[0]) <= 2) and (len(ID_list_cam[1]) <= 2):
        # 记录相同的人
        same_person = []

        # 计算距离矩阵
        Distance_Mat = np.zeros((len(features[0]), len(features[1])), dtype=float)
        for i in range(len(features[0])):
            for j in range(len(features[1])):
                Distance_Mat[i, j] = 10 * np.sqrt(np.sum(np.square(np.array(features[0][i]) - np.array(features[1][j]))))
        # print('-----------------------------------Distance_Mat:', Distance_Mat)

        # 使用匈牙利匹配算法完成匹配
        row, col = linear_sum_assignment(Distance_Mat)
        for i in range(len(row)):
            if Distance_Mat[row[i]][col[i]] < 20:  # 距离小于20视为同一个人
                same_person.append([row[i], col[i] + len(features[0])])

        match_list = np.array(same_person)

        dimGroup = [0, len(features[0]), len(features[0]) + len(features[1])]

        # 获取per的摄像机信息
        match_id_cam_ = []
        for per in range(len(match_list)):  # 以列表中的人为考察对象
            _ = []
            for id in match_list[per]:  # 空间信息匹配列表中存的
                for cam in range(len(dimGroup) - 1):  # 0,1,2
                    if id >= dimGroup[cam] and id < dimGroup[cam + 1]:  # 确定摄像机的ID
                        _.append(cam)
                        break
            match_id_cam_.append(_)

        # 获取per的通道内ID信息
        match_id_ = []
        extend_list = []
        for i in range(len(ID_list_cam)):
            for j in range(len(ID_list_cam[i])):
                extend_list.append(ID_list_cam[i][j])  # 将时序匹配结果展开

        for per in range(len(match_list)):  # 以列表中的人为考察对象
            _ = []
            for id in match_list[per]:  # 空间信息匹配列表中存的
                _.append(extend_list[id])
            match_id_.append(_)

        # 对同摄像头下出现同人情况的处理（MVpose固有缺点）
        for per in range(len(match_id_cam_)):
            for cam in range(cam_number):
                if match_id_cam_[per].count(cam) == 2:  # 若同一个人出现在相同视角下
                    res = [idx for idx, i in enumerate(match_id_cam_[per]) if i == cam]
                    id = res[-1]  # 找到第二个重复的位置
                    match_id_[per][id] = -1  # 禁止更新
        # print(':::::::::::::: Mvpose match_list:', match_list)
        return match_id_cam_, match_id_

    # 只进行Mvpose匹配
    if frame_id % 2 == 0:  # 是5的整倍数，需要进行中央库匹配情况更新

        # 将特征展开成为一个列表
        extend_features = []
        for cam in range(cam_number):
            for feat in features[cam]:
                extend_features.append(feat)
        if len(extend_features) == 0:  # 所有摄像头中没人
            return [], []

        # 获取当前帧roi信息Group：[0,3,5,7]
        dimGroup = [0]
        for i in ID_list_cam:
            dimGroup.append(dimGroup[-1] + len(i))
        reid_affinity = get_reid_affinity(extend_features)
        if len(F) != 0:
            geo_affinity = geometry_affinity(keyp, F, dimGroup=dimGroup)
            match_list = get_match_list(reid_affinity, geo_affinity_mat=geo_affinity, dimGroup=dimGroup)
        else:
            match_list = get_match_list(reid_affinity, dimGroup=dimGroup)

        # 获取per的摄像机信息
        match_id_cam_ = []
        for per in range(len(match_list)):  # 以列表中的人为考察对象
            _ = []
            for id in match_list[per]:  # 空间信息匹配列表中存的
                for cam in range(len(dimGroup) - 1):  # 0,1,2
                    if id >= dimGroup[cam] and id < dimGroup[cam + 1]:  # 确定摄像机的ID
                        _.append(cam)
                        break
            match_id_cam_.append(_)

        # 获取per的通道内ID信息
        match_id_ = []
        extend_list = []
        for i in range(len(ID_list_cam)):
            for j in range(len(ID_list_cam[i])):
                extend_list.append(ID_list_cam[i][j])  # 将时序匹配结果展开

        for per in range(len(match_list)):  # 以列表中的人为考察对象
            _ = []
            for id in match_list[per]:  # 空间信息匹配列表中存的
                _.append(extend_list[id])
            match_id_.append(_)

        # 对同摄像头下出现同人情况的处理（MVpose固有缺点）
        for per in range(len(match_id_cam_)):
            for cam in range(cam_number):
                if match_id_cam_[per].count(cam) == 2:  # 若同一个人出现在相同视角下
                    res = [idx for idx, i in enumerate(match_id_cam_[per]) if i == cam]
                    id = res[-1]  # 找到第二个重复的位置
                    match_id_[per][id] = -1  # 禁止更新
        # print(':::::::::::::: Mvpose match_list:', match_list)
        return match_id_cam_, match_id_

    # 正常4帧，不进行更新
    else:
        return match_id_cam, match_id


def Inter_cam_match_2(tracker, tracker_id, match_id_cam, match_id):
    """根据链接信息生成最终综合ID"""

    # tracker_id_ = copy.deepcopy(tracker_id)  # 将同ID人设为同样颜色
    #
    # for i in range(len(tracker_id_)):
    #     for j in range(len(tracker_id_[i])):
    #         tracker_id_[i][j] = -1  # 先全部置为-1

    for per in range(len(match_id_cam)):  # Mvpose结果下的每个人

        # 若此人出现在了摄像机0中
        if 0 in match_id_cam[per]:
            flag = -1  # 记录首次出现
            for id in range(len(match_id_cam[per])):  # 进入相机编号列表
                cam = match_id_cam[per][id]
                if match_id[per][id] in tracker_id[cam] and flag == -1:  # 记录首次出现
                    flag = match_id[per][id]
                    # tracker_id[cam][tracker_id[cam].index(match_id[per][id])] = flag
                elif match_id[per][id] in tracker_id[cam] and (flag not in tracker_id[cam]):  # 找相同，阻止不同
                    tracker_id[cam][tracker_id[cam].index(match_id[per][id])] = flag
        # 若此人未出现在摄像机1中，不进行ID指派

    return tracker_id
