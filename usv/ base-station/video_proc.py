import cv2
import threading
import numpy as np
import time
from inference import get_roboflow_model

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[2]
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        self.running = True
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while self.running:
            ret, self.last_frame = self.camera.read()
            if not ret:
                self.running = False
                raise Exception("Coulnd't capture video")

class VideoProcessor:
    def __init__(self, host):
        self.model = get_roboflow_model(
            model_id="car_11_4/5", # Roboflow model to use
            api_key="hCR4jqAHhoEQhXUytxsJ"
        )   
        self.cap = cv2.VideoCapture(f"rtsp://{host}:8554/video_stream")
        self.cam_cleaner = CameraBufferCleanerThread(self.cap)
        # self.net = cv2.dnn.readNetFromONNX("config/boat_best.onnx")
        # file = open("config/boat.txt","r")
        # self.classes = file.read().split('\n')
    
    def get_latest_frame(self):
        while self.cam_cleaner.last_frame is None:
            pass

        frame = self.cam_cleaner.last_frame
        return frame

    def init_tracker(self, frame, bbox, label):
        if int(major_ver) < 4 and int(minor_ver) < 3:
            tracker = cv2.cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'CSRT':
                tracker = cv2.TrackerCSRT_create()
            if tracker_type == 'MOSSE':
                tracker = cv2.legacy.TrackerMOSSE_create()
        tracker.init(frame, bbox)
        self.label = label
        return tracker

    def tracking(self, frame):
        ok, bbox = self.tracker.update(frame)
        # Draw bounding box
        if ok:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])	
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(frame, self.label, (x,y-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
        else:
            self.label = "Unknown"
        return ok, bbox

    def detect(self, frame, target):
        height, width, _ = frame.shape
        x_scale = width/640
        y_scale = height/640
        results = self.model.infer(image=frame, confidence=0.5, iou_threshold=0.5)

        best_pred = None
        max_conf = -1

        # Plot image with face bounding box (using opencv)
        for prediction in results[0].predictions:
            if prediction.class_name == target and max_conf < prediction.confidence:
                best_pred = prediction
                max_conf = prediction.confidence
        
        if best_pred:
            cx, cy, w, h = int(best_pred.x), int(best_pred.y), int(best_pred.width), int(best_pred.height)
            x1 = int((cx- w/2)*x_scale)
            y1 = int((cy-h/2)*y_scale)
            wv= int(w * x_scale)
            hv = int(h * y_scale)
            bbox = np.array([x1,y1,wv,hv])
            return True, bbox
        
        return False, None
    
    # def detect(self, frame, target):
    #     # Detect objects using YOLOv5
    #     height, width, _ = frame.shape
    #     blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop=False)
    #     self.net.setInput(blob)
    #     detections = self.net.forward()[0]
                
    #     classes_ids = []
    #     confidences = []
    #     boxes = []
    #     rows = detections.shape[0]

    #     x_scale = width/640
    #     y_scale = height/640

    #     for i in range(rows):
    #         row = detections[i]
    #         confidence = row[4]
    #         if confidence >= 0.4:
    #             classes_score = row[5:]
    #             ind = np.argmax(classes_score)
    #             if classes_score[ind] >= 0.25:
    #                 classes_ids.append(ind)
    #                 confidences.append(confidence)
    #                 cx, cy, w, h = row[:4]
    #                 x1 = int((cx- w/2)*x_scale)
    #                 y1 = int((cy-h/2)*y_scale)
    #                 wv= int(w * x_scale)
    #                 hv = int(h * y_scale)
    #                 box = np.array([x1,y1,wv,hv])
    #                 boxes.append(box)

    #     if len(boxes) > 0:
    #         num_retained_boxes = cv2.dnn.NMSBoxes(boxes,confidences,0.25,0.45)
    #         target_id = -1
    #         max_conf = -1 
    #         for i in num_retained_boxes:
    #             if self.classes[classes_ids[i]] == target and max_conf < confidences[i]:
    #                 target_id = i
    #                 max_conf = confidences[i]
                        
    #         if target_id > -1:
    #             bbox = boxes[target_id]
    #             return True, bbox
    #     return False, None 

    def close(self):
        self.cam_cleaner.running = False
        time.sleep(1)
        self.cap.release()
    