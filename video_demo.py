#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
from face_detection.face_detect_interface import FaceDetector, box_transform
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts, predict_dense, parse_roi_box_from_bbox
from utils.estimate_pose import parse_pose
from utils.cv_plot import plot_pose_box
from utils.render import get_depths_image, cget_depths_image, cpncc
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE = 120


def draw_landmarks(image, landmarks):
    """Draw landmarks using cv2"""
    for index, landmark in enumerate(landmarks):
        pt1 = (landmark[0], landmark[1])
        cv2.circle(image, pt1, 1, (255, 255, 255), 2)
        # cv2.putText(image, str(index), pt1, 1, 0.5, (50, 50, 150))
    plot_line = lambda i1, i2: cv2.line(image,
                                        (landmarks[i1][0],
                                         landmarks[i1][1]),
                                        (landmarks[i2][0],
                                         landmarks[i2][1]),
                                        (255, 255, 255), 1)
    close_point_list = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
    for ind in range(len(close_point_list) - 1):
        l, r = close_point_list[ind], close_point_list[ind + 1]
        # 根据部位绘制关键点连线
        for index in range(l, r-1):
            plot_line(index, index+1)
        # 将眼部, 嘴部连线闭合
        plot_line(41, 36)  # 左眼
        plot_line(47, 42)  # 右眼
        plot_line(59, 48)  # 外唇
        plot_line(67, 60)  # 内唇


def pts2landmarks(pts):
    landmarks = [[pts[0][index], pts[1][index]] for index in range(68)]
    return landmarks


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    face_detector = FaceDetector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    video = cv2.VideoCapture()
    auto_play_flag = False
    decay_time = 1 if auto_play_flag else 0
    if not video.open(args.video_path):
        print("can not open camera!")
        return
    while True:
        _, img_ori = video.read()
        if img_ori is None:
            break
        rects = face_detector.detect(img_ori)
        rects = box_transform(rects, img_ori.shape[1], img_ori.shape[0])
        pts_res = []
        vertices_lst = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        for rect in rects:
            bbox = [rect[1], rect[2], rect[3], rect[4]]
            roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)
            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            for pts in pts_res:
                landmarks = pts2landmarks(pts)
                draw_landmarks(img_ori, landmarks)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)
            vertices = predict_dense(param, roi_box)
            vertices_lst.append(vertices)
        if args.show_pose:
            # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
            img_pose = plot_pose_box(img_ori, Ps, pts_res)
            img_ori = img_pose
        if args.show_depth:
            # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
            # img_ori = depths_img
            img_ori = (img_ori + 0.5 * np.tile(depths_img, (3, 1, 1)).transpose([1, 2, 0])) / 255.0
        if args.show_pncc:
            pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
            # pncc_feature = pncc_feature[:, :, ::-1]  # swap RGB -> BGR
            # img_ori = np.asarray(img_ori + 0.5 * pncc_feature, dtype=np.uint8())
            img_ori = (img_ori + 0.5 * pncc_feature)/255.0
            # print(pncc_feature)
            # exit(1)
        cv2.imshow("video", img_ori)
        key = cv2.waitKey(decay_time)
        if key == 32:
            auto_play_flag = not auto_play_flag
            decay_time = 1 if auto_play_flag else 0
        if key == 27:
            break
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-v', '--video_path', default=0, type=str, help='video path， default use camera')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--show_pose', default='false', type=str2bool)
    parser.add_argument('--show_depth', default='false', type=str2bool)
    parser.add_argument('--show_pncc', default='false', type=str2bool)

    args = parser.parse_args()
    print(args)
    main(args)
