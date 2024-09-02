from ultralytics import YOLO
import torch
import numpy as np
import cv2
import pprint, time

''' print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
#torch.cuda.set_device(0) # Set to your desired GPU number
print("torch.cuda.is_available()",torch.cuda.is_available()) '''

default_models = ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt"] 
pose_detector = YOLO( default_models[0], verbose= True)    
unknown_detector = YOLO("bestest.pt", verbose= True)    

def detect_and_return_results_pose(cv2_frame:np.ndarray = None, yolo_object:YOLO = None, show=False, save=False, threshold_confidence = 0.5):
    KEYPOINT_NAMES = ["nose", "right_eye", "left_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow" ,"right_elbow","left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
    formatted_detections = []
    detections = yolo_object(cv2_frame, task = "pose", verbose= True, show = show, save = save)[0]
    time.sleep(10)
    for detection in detections:
        boxes = detection.boxes
        box_cls_no = int(boxes.cls.cpu().numpy()[0])
        box_cls_name = yolo_object.names[box_cls_no]
        box_conf = boxes.conf.cpu().numpy()[0]
        box_xyxyn = boxes.xyxyn.cpu().numpy()[0]
        
        if box_cls_name not in ["person"] and box_conf < threshold_confidence:
            continue

        key_points = detection.keypoints
        keypoint_confs = key_points.conf.cpu().numpy()[0]
        keypoints_xyn = key_points.xyn.cpu().numpy()[0]

        normalized_keypoints_dict = {}  
        for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
            keypoint_xn = keypoints_xyn[keypoint_index][0]
            keypoint_yn = keypoints_xyn[keypoint_index][1]
            keypoint_conf = keypoint_confs[keypoint_index] 
            if keypoint_xn == 0 and keypoint_yn == 0: #if the keypoint is not detected
                #But this is also a prediction. Thus the confidence should not be set to zero. negative values are used to indicate that the keypoint is not detected
                keypoint_conf = -keypoint_conf
            normalized_keypoints_dict[keypoint_name] = [keypoint_xn, keypoint_yn, keypoint_conf]
    
        prediction_dict = { "class_name": box_cls_name, "confidence": box_conf, "bbox": [box_xyxyn[0], box_xyxyn[1], box_xyxyn[2], box_xyxyn[3]], "keypoints": normalized_keypoints_dict}
        formatted_detections.append(prediction_dict)    
    return formatted_detections

def detect_and_return_results_detect(cv2_frame:np.ndarray = None, yolo_object:YOLO = None, show=False, threshold_confidence = 0.0):
    
    formatted_detections = []
    detections = yolo_object(cv2_frame, task = "detect", verbose= True, show = show)[0]
    time.sleep(25)

    for detection in detections:
        boxes = detection.boxes
        box_cls_no = int(boxes.cls.cpu().numpy()[0])
        box_cls_name = yolo_object.names[box_cls_no]
        box_conf = boxes.conf.cpu().numpy()[0]
        box_xyxyn = boxes.xyxyn.cpu().numpy()[0]
        
        considered_classes = ["person"]
        if box_cls_name not in considered_classes and box_conf < threshold_confidence:
            continue

        prediction_dict = { "class_name": box_cls_name, "confidence": box_conf, "bbox": [box_xyxyn[0], box_xyxyn[1], box_xyxyn[2], box_xyxyn[3]]}
        formatted_detections.append(prediction_dict)
    
    return formatted_detections

if __name__ == "_main_":
    
    ''' cv2_frame = cv2.imread("santiye.jpeg")
    formatted_detection_results = detect_and_return_results_pose(cv2_frame= cv2_frame, yolo_object= pose_detector, threshold_confidence= 0.5, show = True)
    pprint.pprint(formatted_detection_results) '''

    cv2_frame = cv2.imread("zoomed.jpeg")
    formatted_detection_results = detect_and_return_results_pose(cv2_frame= cv2_frame, yolo_object= pose_detector, threshold_confidence= 0.5, show = False, save = False)
    pprint.pprint(formatted_detection_results)

    ''' cv2_frame = cv2.imread("construction.png")
    formatted_detection_results = detect_and_return_results_detect(cv2_frame= cv2_frame, yolo_object= unknown_detector, threshold_confidence= 0.0, show = True)
    pprint.pprint(formatted_detection_results) '''