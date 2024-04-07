import numpy as np
import threading
import time
import cv2
from car import Car

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[5]

def init_tracker(frame, bbox):
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
    return tracker

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()

def follow_object(Ab, bbox, width):
    x, w  = int(bbox[0]), int(bbox[2])
    cx = int(x + w // 2)
    if cx < width // 2 - 50: 
        Ab.left()
        print("tracking: Left")
    elif cx > width // 2 + 50: 
        Ab.right() 
        print("tracking: Right")
    else:  
        Ab.forward()
        print("tracking: Forward")

class ObjectDetectingThread(threading.Thread):
    def __init__(self, Ab, detecting=True, name='object-detecting-thread'):
        self.Ab = Ab
        self.detecting = detecting
        self.tracking = False
        super(ObjectDetectingThread, self).__init__(name=name)
        self.start()

    def run(self):
        cap = cv2.VideoCapture(0)
        cam_cleaner = CameraBufferCleanerThread(cap)
        net = cv2.dnn.readNetFromONNX("best.onnx")
        
        classes = ['obstacle']
        bbox = None
        # obj_label = ""

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            if cam_cleaner.last_frame is None:
                continue    

            frame = cam_cleaner.last_frame

            if self.tracking:
                ok, bbox = tracker.update(frame)
                if ok:
                    # x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    # cx = int(x + w // 2)
                    # cy = int(y + h // 2)
                    #cv2.line(frame, (int(cx), int(cy)), (width // 2, height), (255,255,255), 2)
                    
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
                    #cv2.putText(frame, text, (x,y-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
                    self.car.make_decision(follow_object, bbox, width)
                else :
                    self.tracking = False
                    self.car.stop()
            else:
                blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                detections = net.forward()[0]
                
                classes_ids = []
                confidences = []
                boxes = []
                rows = detections.shape[0]

                x_scale = width/640
                y_scale = height/640

                for i in range(rows):
                    row = detections[i]
                    confidence = row[4]
                    if confidence > 0.5:
                        classes_score = row[5:]
                        ind = np.argmax(classes_score)
                        if classes_score[ind] > 0.5:
                            classes_ids.append(ind)
                            confidences.append(confidence)
                            cx, cy, w, h = row[:4]
                            x1 = int((cx- w/2)*x_scale)
                            y1 = int((cy-h/2)*y_scale)
                            wv= int(w * x_scale)
                            hv = int(h * y_scale)
                            box = np.array([x1,y1,wv,hv])
                            boxes.append(box)

                if len(boxes) > 0:
                    num_retained_boxes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
                    for i in num_retained_boxes:
                        if classes[classes_ids[i]] == 'obstacle':
                            bbox = boxes[i]
                            self.tracking = True
                            tracker = init_tracker(frame, bbox)
                            follow_object(self.Ab, bbox, width)
                            break
            # Display frame with bounding box
            #cv2.imshow("Frame", frame)

            #if cv2.waitKey(1) & 0xFF == ord('q'): 
                #break
