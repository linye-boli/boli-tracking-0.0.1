# -*- coding: utf-8 -*-
import torch
import json
import os
import zipfile
import time
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from opt import opt
from PIL import Image
import torchvision.transforms as transforms

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
areaThres = 0  # 40 * 40.5
alpha = 0.1


# pool = ThreadPool(4)


def pose_nms(bboxes, bbox_scores, pose_preds, pose_scores, single_height, orig_img, output_length):
    '''
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n,)
    pose_preds:     pose locations list (n, 17, 2)
    pose_scores:    pose scores list    (n, 17, 1)
    '''
    # global ori_pose_preds, ori_pose_scores, ref_dists

    pose_scores[pose_scores == 0] = 1e-5

    final_result = []

    ori_bbox_scores = bbox_scores.clone()
    ori_pose_preds = pose_preds.clone()
    ori_pose_scores = pose_scores.clone()
    ori_bboxes = bboxes.clone()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(dim=1)

    human_ids = np.arange(nsamples)  # ROI的个数
    # Do pPose-NMS
    pick = []  # 保存经过NMS之后的ROI
    merge_ids = []

    while (human_scores.shape[0] != 0):
        # Pick the one with highest score
        pick_id = torch.argmax(human_scores)
        pick.append(human_ids[pick_id])  # 选取关键点分数最高的ROI——ID
        # num_visPart = torch.sum(pose_scores[pick_id] > 0.2)

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[pick_id]]
        simi = get_parametric_distance(pick_id, pose_preds, pose_scores, ref_dist)
        num_match_keypoints = PCK_match(pose_preds[pick_id], pose_preds, ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[(simi > gamma) | (num_match_keypoints >= matchThreds)]

        if delete_ids.shape[0] == 0:
            delete_ids = pick_id
        # else:
        #    delete_ids = torch.from_numpy(delete_ids)
        # 删除掉不满足NMS的ROI，然后继续执行while循环，直到为空
        merge_ids.append(human_ids[delete_ids])
        pose_preds = np.delete(pose_preds, delete_ids, axis=0)
        pose_scores = np.delete(pose_scores, delete_ids, axis=0)
        human_ids = np.delete(human_ids, delete_ids)
        human_scores = np.delete(human_scores, delete_ids, axis=0)
        bbox_scores = np.delete(bbox_scores, delete_ids, axis=0)

    assert len(merge_ids) == len(pick)
    # 在原始数据中取出来符合条件的ROI，pick就是下标列表
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    bboxes_pick = ori_bboxes[pick]
    # final_result = pool.map(filter_result, zip(scores_pick, merge_ids, preds_pick, pick, bbox_scores_pick))
    # final_result = [item for item in final_result if item is not None]

    # prepare to separate frame
    boxes_1 = [];
    boxes_2 = []
    scores_1 = [];
    scores_2 = []
    roi_1 = [];
    roi_2 = []
    orig_img_1 = orig_img[:single_height, :, :]
    orig_img_2 = orig_img[single_height:, :, :]
    kp_1 = [];
    kp_2 = []
    kp_score_1 = [];
    kp_score_2 = []
    output_l = output_length if len(pick) > output_length else len(pick)
    for j in range(output_l):
        ids = np.arange(17)
        max_score = torch.max(scores_pick[j, ids, 0])

        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j], ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick[j]])

        max_score = torch.max(merge_score[ids])
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])

        if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < areaThres):
            continue

        kp_score = merge_score
        proposal_score = torch.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score)  # 总的关键点得分，暂时没用

        # 暂时用flag 表示属于哪个摄像头，以后再根据下游接口修改
        # 校正关键点超界,分离两摄像头数据
        kp = merge_pose - 0.3
        data_resize = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3)
        ])
        x1, y1, x2, y2 = bboxes_pick[j]
        if y1 < single_height and y2 <= single_height:
            boxes_1.append(bboxes_pick[j])
            scores_1.append(bbox_scores_pick[j])
            # roi
            sn = scores_1[-1].numpy()
            # if(sn > 0.85):  
            bn = boxes_1[-1].numpy()
            roi_temp = orig_img_1[int(bn[1]):int(bn[3]), int(bn[0]):int(bn[2])]
            roi_temp = Image.fromarray(roi_temp.astype('uint8')).convert('RGB')
            roi_temp = data_resize(roi_temp)
            roi_1.append(roi_temp)
            # keypoint
            kp = merge_pose - 0.3
            kp_1.append(kp)
            kp_score_1.append(kp_score)

        elif y1 < single_height and y2 > single_height:
            if (y1 + y2) / 2 < single_height:  # 下边框落在第二图中
                boxes_1.append(bboxes_pick[j])
                scores_1.append(bbox_scores_pick[j])
                boxes_1[-1][3] = single_height  # 校正
                # roi
                sn = scores_1[-1].numpy()
                # if(sn > 0.85):  
                bn = boxes_1[-1].numpy()
                roi_temp = orig_img_1[int(bn[1]):int(bn[3]), int(bn[0]):int(bn[2])]
                roi_temp = Image.fromarray(roi_temp.astype('uint8')).convert('RGB')
                roi_temp = data_resize(roi_temp)
                roi_1.append(roi_temp)
                # keypoints
                if ymax > single_height:
                    for k in range(len(merge_pose)):
                        if merge_pose[k][1] > single_height:
                            merge_pose[k][1] = single_height
                        else:
                            continue
                kp = merge_pose - 0.3
                kp_1.append(kp)
                kp_score_1.append(kp_score)
            else:  # 第二图的上边框落在第一图中
                boxes_2.append(bboxes_pick[j])
                scores_2.append(bbox_scores_pick[j])
                boxes_2[-1][1] = single_height  # 校正
                boxes_2[-1][1] -= single_height
                boxes_2[-1][3] -= single_height
                # roi
                sn = scores_2[-1].numpy()
                # if(sn > 0.85):  
                bn = boxes_2[-1].numpy()
                roi_temp = orig_img_2[int(bn[1]):int(bn[3]), int(bn[0]):int(bn[2])]
                roi_temp = Image.fromarray(roi_temp.astype('uint8')).convert('RGB')
                roi_temp = data_resize(roi_temp)
                roi_2.append(roi_temp)
                # keypoints
                if ymin < single_height:
                    for k in range(len(merge_pose)):
                        if merge_pose[k][1] < single_height:
                            merge_pose[k][1] = single_height
                        else:
                            continue
                merge_pose[:, 1] -= single_height
                kp = merge_pose - 0.3
                kp_2.append(kp)
                kp_score_2.append(kp_score)
        else:
            boxes_2.append(bboxes_pick[j])
            scores_2.append(bbox_scores_pick[j])
            boxes_2[-1][3] -= single_height
            boxes_2[-1][1] -= single_height
            # roi
            sn = scores_2[-1].numpy()
            # if(sn > 0.85):  
            bn = boxes_2[-1].numpy()
            roi_temp = orig_img_2[int(bn[1]):int(bn[3]), int(bn[0]):int(bn[2])]
            roi_temp = Image.fromarray(roi_temp.astype('uint8')).convert('RGB')
            roi_temp = data_resize(roi_temp)
            roi_2.append(roi_temp)
            # keypoints
            merge_pose[:, 1] -= single_height
            kp = merge_pose - 0.3
            kp_2.append(kp)
            kp_score_2.append(kp_score)

        # final_result.append({
        #     'keypoints': merge_pose - 0.3,
        #     'kp_score': merge_score,
        #     'proposal_score': torch.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score), 
        # })
    box = [boxes_1, boxes_2]
    box_s = [scores_1, scores_2]
    roi = [roi_1, roi_2]
    kp = [kp_1, kp_2]
    kp_s = [kp_score_1, kp_score_2]

    return box, box_s, roi, kp, kp_s


def filter_result(args):
    score_pick, merge_id, pred_pick, pick, bbox_score_pick = args
    global ori_pose_preds, ori_pose_scores, ref_dists
    ids = np.arange(17)
    max_score = torch.max(score_pick[ids, 0])

    if max_score < scoreThreds:
        return None

    # Merge poses
    merge_pose, merge_score = p_merge_fast(
        pred_pick, ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick])

    max_score = torch.max(merge_score[ids])
    if max_score < scoreThreds:
        return None

    xmax = max(merge_pose[:, 0])
    xmin = min(merge_pose[:, 0])
    ymax = max(merge_pose[:, 1])
    ymin = min(merge_pose[:, 1])

    if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < 40 * 40.5):
        return None

    return {
        'keypoints': merge_pose - 0.3,
        'kp_score': merge_score,
        'proposal_score': torch.mean(merge_score) + bbox_score_pick + 1.25 * max(merge_score)
    }


def p_merge(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    '''
    dist = torch.sqrt(torch.sum(
        torch.pow(ref_pose[np.newaxis, :] - cluster_preds, 2),
        dim=2
    ))  # [n, 17]

    kp_num = 17
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = torch.zeros(kp_num, 2)
    final_score = torch.zeros(kp_num)

    if cluster_preds.dim() == 2:
        cluster_preds.unsqueeze_(0)
        cluster_scores.unsqueeze_(0)
    if mask.dim() == 1:
        mask.unsqueeze_(0)

    for i in range(kp_num):
        cluster_joint_scores = cluster_scores[:, i][mask[:, i]]  # [k, 1]
        cluster_joint_location = cluster_preds[:, i, :][mask[:, i].unsqueeze(
            -1).repeat(1, 2)].view((torch.sum(mask[:, i]), -1))

        # Get an normalized score
        normed_scores = cluster_joint_scores / torch.sum(cluster_joint_scores)

        # Merge poses by a weighted sum
        final_pose[i, 0] = torch.dot(cluster_joint_location[:, 0], normed_scores.squeeze(-1))
        final_pose[i, 1] = torch.dot(cluster_joint_location[:, 1], normed_scores.squeeze(-1))

        final_score[i] = torch.dot(cluster_joint_scores.transpose(0, 1).squeeze(0), normed_scores.squeeze(-1))

    return final_pose, final_score


def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    '''
    dist = torch.sqrt(torch.sum(
        torch.pow(ref_pose[np.newaxis, :] - cluster_preds, 2),
        dim=2
    ))

    kp_num = 17
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = torch.zeros(kp_num, 2)
    final_score = torch.zeros(kp_num)

    if cluster_preds.dim() == 2:
        cluster_preds.unsqueeze_(0)
        cluster_scores.unsqueeze_(0)
    if mask.dim() == 1:
        mask.unsqueeze_(0)

    # Weighted Merge
    masked_scores = cluster_scores.mul(mask.float().unsqueeze(-1))
    normed_scores = masked_scores / torch.sum(masked_scores, dim=0)

    final_pose = torch.mul(cluster_preds, normed_scores.repeat(1, 1, 2)).sum(dim=0)
    final_score = torch.mul(masked_scores, normed_scores).sum(dim=0)
    return final_pose, final_score


def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_preds[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    mask = (dist <= 1)

    # Define a keypoints distance
    score_dists = torch.zeros(all_preds.shape[0], 17)
    keypoint_scores.squeeze_()
    if keypoint_scores.dim() == 1:
        keypoint_scores.unsqueeze_(0)
    if pred_scores.dim() == 1:
        pred_scores.unsqueeze_(1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = pred_scores.repeat(1, all_preds.shape[0]).transpose(0, 1)

    score_dists[mask] = torch.tanh(pred_scores[mask] / delta1) * torch.tanh(keypoint_scores[mask] / delta1)

    point_dist = torch.exp((-1) * dist / delta2)
    final_dist = torch.sum(score_dists, dim=1) + mu * torch.sum(point_dist, dim=1)

    return final_dist


def PCK_match(pick_pred, all_preds, ref_dist):
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_pred[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = torch.sum(
        dist / ref_dist <= 1,
        dim=1
    )

    return num_match_keypoints


def write_json(all_results, outputpath, for_eval=False):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    form = opt.format
    json_results = []
    json_results_cmu = {}
    for im_res in all_results:
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            result = {}
            if for_eval:
                result['image_id'] = int(im_name.split('/')[-1].split('.')[0].split('_')[-1])
            else:
                result['image_id'] = im_name.split('/')[-1]
            result['category_id'] = 1

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)

            if form == 'cmu':  # the form of CMU-Pose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']] = {}
                    json_results_cmu[result['image_id']]['version'] = "AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['bodies'] = []
                tmp = {'joints': []}
                result['keypoints'].append((result['keypoints'][15] + result['keypoints'][18]) / 2)
                result['keypoints'].append((result['keypoints'][16] + result['keypoints'][19]) / 2)
                result['keypoints'].append((result['keypoints'][17] + result['keypoints'][20]) / 2)
                indexarr = [0, 51, 18, 24, 30, 15, 21, 27, 36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
                for i in indexarr:
                    tmp['joints'].append(result['keypoints'][i])
                    tmp['joints'].append(result['keypoints'][i + 1])
                    tmp['joints'].append(result['keypoints'][i + 2])
                json_results_cmu[result['image_id']]['bodies'].append(tmp)
            elif form == 'open':  # the form of OpenPose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']] = {}
                    json_results_cmu[result['image_id']]['version'] = "AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['people'] = []
                tmp = {'pose_keypoints_2d': []}
                result['keypoints'].append((result['keypoints'][15] + result['keypoints'][18]) / 2)
                result['keypoints'].append((result['keypoints'][16] + result['keypoints'][19]) / 2)
                result['keypoints'].append((result['keypoints'][17] + result['keypoints'][20]) / 2)
                indexarr = [0, 51, 18, 24, 30, 15, 21, 27, 36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
                for i in indexarr:
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i + 1])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i + 2])
                json_results_cmu[result['image_id']]['people'].append(tmp)
            else:
                json_results.append(result)

    if form == 'cmu':  # the form of CMU-Pose
        with open(os.path.join(outputpath, 'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath, 'sep-json')):
                os.mkdir(os.path.join(outputpath, 'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath, 'sep-json', name.split('.')[0] + '.json'), 'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    elif form == 'open':  # the form of OpenPose
        with open(os.path.join(outputpath, 'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath, 'sep-json')):
                os.mkdir(os.path.join(outputpath, 'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath, 'sep-json', name.split('.')[0] + '.json'), 'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    else:
        with open(os.path.join(outputpath, 'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results))
