# -*- coding: utf-8 -*-

import numpy as np
import math
from itertools import product
from face_detection.face_detect_interface.utils import config


class CDefaultBoxes(object):
    def __init__(self):
        self._input_image_height = config.fixed_input_height
        self._input_image_width = config.fixed_input_width
        self._default_boxes_param = config.default_boxes_params
        self._scales = list()
        self._create_scales()

        assert len(self._default_boxes_param['feature_maps_size']) == len(self._scales)
        assert len(self._default_boxes_param['steps']) == len(self._scales)
        assert len(self._default_boxes_param['aspect_ratios']) == len(self._scales)

    def _create_scales(self):
        min_scale = self._default_boxes_param['s_min']
        max_scale = self._default_boxes_param['s_max']
        feature_maps_count = len(self._default_boxes_param['feature_maps_size'])
        for i in range(feature_maps_count):
            scale = min_scale + (max_scale - min_scale) * i / (feature_maps_count - 1)
            self._scales.append(scale)

    def create_default_boxes(self):
        default_boxes = list()
        for k, f in enumerate(self._default_boxes_param['feature_maps_size']):
            assert len(f) == 2
            for y, x in product(list(range(f[0])), list(range(f[1]))):
                # corresponding coordinates in input image
                cx = (x + 0.5) * self._default_boxes_param['steps'][k] / self._input_image_width
                cy = (y + 0.5) * self._default_boxes_param['steps'][k] / self._input_image_height
                for ar in self._default_boxes_param['aspect_ratios'][k]:
                    default_boxes += [cx, cy, self._scales[k]*math.sqrt(ar), self._scales[k]/math.sqrt(ar)]
        output = np.array(default_boxes, dtype=np.float32)
        output = np.reshape(output, newshape=(-1, 4))
        output = np.clip(output, a_min=0, a_max=1)
        return output

    def __call__(self, *args, **kwargs):
        return self.create_default_boxes()


# def main():
#     default_boxes = CDefaultBoxes()()
#     print(default_boxes.dtype)
#     print(default_boxes.shape)
#     default_boxes = default_boxes.reshape(default_boxes.shape[0]*default_boxes.shape[1])
#     file = open("default_boxes.txt", "w")
#     for idx, point in enumerate(default_boxes):
#         file.write(str(point) + ' , ')
#         if (idx+1) % 32 == 0:
#             file.write('\n')


# if __name__ == "__main__":
#     main()
