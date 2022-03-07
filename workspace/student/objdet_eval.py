# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools

###[S4-EX1]
def calculate_iou(gt_bbox_points, pred_bbox_points):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    gt_bbox_points = np.array(gt_bbox_points)
    gt_bbox_x1 = np.min(gt_bbox_points[:,0])
    gt_bbox_y1 = np.min(gt_bbox_points[:,1])
    gt_bbox_x2 = np.max(gt_bbox_points[:,0])
    gt_bbox_y2 = np.max(gt_bbox_points[:,1])
    gt_bbox = [gt_bbox_x1, gt_bbox_y1, gt_bbox_x2, gt_bbox_y2]
    
    pred_bbox_points = np.array(pred_bbox_points)
    pred_bbox_x1 = np.min(pred_bbox_points[:,0])
    pred_bbox_y1 = np.min(pred_bbox_points[:,1])
    pred_bbox_x2 = np.max(pred_bbox_points[:,0])
    pred_bbox_y2 = np.max(pred_bbox_points[:,1])
    pred_bbox = [pred_bbox_x1, pred_bbox_y1, pred_bbox_x2, pred_bbox_y2]
    
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
     # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # exclude all labels from statistics which are not considered valid
            
            # compute intersection over union (iou) and distance between centers
            # ユニオン（iou）上の交差と中心間の距離を計算します

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            ## ステップ1：現在のラベルバウンディングボックスの四隅を抽出します
            bbox = label.box
            bbox_corners = tools.compute_box_corners(bbox.center_x, bbox.center_y, bbox.width, bbox.length, bbox.heading)   # 参考：label_corners
            label_area = Polygon(bbox_corners)                                                                              # 参考：label_area = Polygon(label_corners)
            
            ## step 2 : loop over all detected objects
            ## ステップ2：検出されたすべてのオブジェクトをループします
            for a_detection in detections:                                                                                  # 参考：det

                ## step 3 : extract the four corners of the current detection
                ## ステップ3：現在の検出の四隅を抽出します
                _, obj_x, obj_y, obj_z, obj_h, obj_w, obj_l, obj_yaw = a_detection                                          # 参考：_, x, y, z, _, w, l, yaw = det
                obj_box_corners = tools.compute_box_corners(obj_x, obj_y, obj_w, obj_l, obj_yaw)                            # 参考：det_corners
                det_area = Polygon(obj_box_corners)                                                                         # 参考：det_area = Polygon(det_corners)
                
                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                ## ステップ4：x、y、zのラベルと検出バウンディングボックス間の中心距離を計算します
                dist_x = bbox.center_x - obj_x
                dist_y = bbox.center_y - obj_y
                dist_z = bbox.center_z - obj_z
                
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                ## ステップ5：ラベルと検出バウンディングボックス間の結合上の交差（IOU）を計算します
                ##### corners = [fl,rl,rr,fr]
                print("bbox_corners",bbox_corners)
                print("obj_box_corners",obj_box_corners)
#                iou, _ = calculate_iou(np.array(obj_box_corners), np.array(bbox_corners))
                intersec = label_area.intersection(det_area)                                                                # 参考
                union = label_area.union(det_area)                                                                          # 参考
                iou = intersec.area / union.area                                                                            # 参考
                print("iou",iou)
  
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                ## ステップ6：IOUがmin_iouしきい値を超えた場合は、[iou、dist_x、dist_y、dist_z]をmatches_lab_detに保存し、TPカウントを増やします
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                    true_positives += 1

                
            #######
            ####### ID_S4_EX1 END #######     
            
        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])


    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    # 適合率/再現率の正と負を計算する
    
    ## step 1 : compute the total number of positives present in the scene
    ## ステップ1：シーンに存在するポジティブの総数を計算します
#    all_positives = labels_valid.sum()
#    print("all_positives", all_positives)
    all_positives = len([p for p in labels_valid if p])
    
    ## step 2 : compute the number of false negatives
    ## ステップ2：偽陰性の数を計算する
    true_positives = len(ious)
    false_negatives = all_positives - true_positives

    ## step 3 : compute the number of false positives
    ## ステップ3：誤検知の数を計算する
    false_positives = len(detections) - true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all, configs_det):     #[S4_EX3 addtion]

    # extract elements
    # 要素を抽出する
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')
    
    configs_det.use_labels_as_objects = True    #[S4_EX3 addtion]
        
    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    ## ステップ1：ポジティブ、トゥルーポジティブ、フォールスネガティブ、フォールスポジティブの総数を抽出します
    print("pos_negs",pos_negs)    
    all_positives, true_positives, false_negatives, false_positives = np.array(pos_negs).sum(axis=0)

    ## step 2 : compute precision
    ## ステップ2：精度を計算する
    precision = true_positives / (true_positives + false_positives)

    ## step 3 : compute recall 
    ## ステップ3：リコールの計算
    recall = true_positives / (true_positives + false_negatives)

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()