import os
from face_detect_interface import FaceDetector, box_transform, draw_detection_rects
import cv2
working_root = os.path.split(os.path.realpath(__file__))[0]
os.chdir(working_root)

if __name__ == '__main__':
    detector = FaceDetector()
    image_path = os.path.join(os.getcwd(), "../samples/emma_input.jpg")
    image = cv2.imread(image_path, 1)
    bounding_boxes = detector.detect(image)
    image = cv2.resize(image, dsize=(640, 576))
    draw_detection_rects(image, bounding_boxes)
    cv2.imshow("image", image)
    cv2.waitKey()
    print(bounding_boxes)
