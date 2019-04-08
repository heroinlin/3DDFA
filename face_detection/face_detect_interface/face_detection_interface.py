import os
import cv2
import numpy as np
import torch
from .read_data.default_boxes import CDefaultBoxes
from .utils.nms import py_cpu_nms
from .utils.box_utils import decode
root_dir = os.path.split(os.path.realpath(__file__))[0]


def box_transform(bounding_boxes, width, height):
    """
    bounding_boxes  [[score, box],[score, box]]
    box框结果值域由[0,1],[0,1] 转化为[0,width]和[0,height]
    """
    for i in range(len(bounding_boxes)):
        x1 = float(bounding_boxes[i][1])
        y1 = float(bounding_boxes[i][2])
        x2 = float(bounding_boxes[i][3])
        y2 = float(bounding_boxes[i][4])
        bounding_boxes[i][1] = x1 * width
        bounding_boxes[i][2] = y1 * height
        bounding_boxes[i][3] = x2 * width
        bounding_boxes[i][4] = y2 * height
    return bounding_boxes


def post_process(scores, boxes, detect_threshold=0.7, nms_threshold=0.3):
    indices = np.where(scores > detect_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    keep = py_cpu_nms(boxes, scores, nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    return scores, boxes


def draw_detection_rects(image, detection_rects, color=(0, 255, 0)):
    for rect in detection_rects:
        cv2.rectangle(image,
                      (int(rect[1] * image.shape[1]), int(rect[2] * image.shape[0])),
                      (int(rect[3] * image.shape[1]), int(rect[4] * image.shape[0])),
                      color,
                      thickness=2)
        cv2.putText(image, f"{rect[0]:.03f}",
                    (int(rect[1] * image.shape[1]), int(rect[2] * image.shape[0])), 1, 1, (255, 0, 255))


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for name, module1 in module._modules.items():
            recursion_change_bn(module1)
    return module


class FaceDetector(object):
    def __init__(self, checkpoint_file_path=None, detect_threshold=0.7, nms_threshold=0.3, device=None):
        super(FaceDetector, self).__init__()
        self.checkpoint_file_path = checkpoint_file_path
        self.default_boxes = None
        self.transforms = None
        self.model = None
        self.device = device
        self.model_loader()
        self.config = {
            "width": 160,
            "height": 160,
            "mean":  [0.486, 0.459, 0.408],
            "stddev": [0.229, 0.224, 0.225],
            "detect_threshold": detect_threshold,
            "nms_threshold": nms_threshold
        }

    def model_loader(self):
        if torch.__version__ < "1.0.0":
            print("Pytorch version is not  1.0.0, please check it!")
            exit(-1)
        if self.checkpoint_file_path is None:
            self.checkpoint_file_path = os.path.join(root_dir, "models/mobilenet_v2_0.25_43_0.1162_jit.pth")
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # check_point = torch.load(self.checkpoint_file_path, map_location=self.device)
        # self.model = check_point['net'].to(self.device)
        self.model = torch.jit.load(self.checkpoint_file_path, map_location=self.device)
        # 如果模型为pytorch0.3.1版本，需要以下代码添加BN内的参数
        # for _, module in self.model._modules.items():
        #     recursion_change_bn(self.model)
        self.model.eval()
        self.default_boxes = CDefaultBoxes().create_default_boxes()

    def set_config(self, key, value):
        self.config[key] = value

    def detect(self, input_image):
        input_image = cv2.resize(input_image, dsize=(self.config['width'], self.config['height']))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = (np.array(input_image, dtype=np.float32) / 255 - self.config['mean']) / self.config['stddev']
        input_image = torch.from_numpy(input_image.transpose([2, 0, 1])).float()
        input_image = torch.unsqueeze(input_image, dim=0)
        input_image = torch.autograd.Variable(input_image.to(self.device))

        conf_predict, loc_predict = self.model(input_image)
        conf_predict = conf_predict.view(-1).sigmoid().data.cpu().numpy()
        loc_predict = loc_predict.view(-1, 4).data.cpu().numpy()
        loc_predict = decode(loc_predict, self.default_boxes)
        scores, boxes = post_process(conf_predict, loc_predict,
                                     self.config.get("detect_threshold"), self.config.get("nms_threshold"))
        scores = scores.reshape(len(scores), 1)
        bounding_boxes = np.hstack((scores, boxes))
        return bounding_boxes


