# -*- coding: utf-8 -*-

fixed_input_height, fixed_input_width = 160, 160

default_boxes_params = {
    'feature_maps_size': [[20, 20], [10, 10], [5, 5], [3, 3], [1, 1]],
    # 'feature_maps_size': [[9, 10], [5, 5], [3, 3], [1, 1]],
    'steps': [8, 16, 32, 64, 128],
    # 'steps': [8, 16, 32, 64],
    's_min': 0.15,
    's_max': 0.9,
    # height width ratio
    'aspect_ratios': [[1], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]]
    # 'aspect_ratios': [[1], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]]
}

matcher_param = {
    'threshold': 0.4,
    'encoded_variance': [0.1, 0.2]
}

classes_number = 1


eval_params = {
    'detect_threshold': 0.70,
    'nms_threshold': 0.3,
    'recall_threshold': 0.5
}
