import RPi.GPIO as GPIO
from AlphaBot import AlphaBot
import pickle
import numpy as np
import threading
import time
import cv2

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
            _, self.last_frame = self.camera.read()

def follow_object(bbox, width):
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

def detect():
    global tracking 

    while True:
        if cam_cleaner.last_frame is None:
            time.sleep(0.01)
            continue    

        frame = cam_cleaner.last_frame

        if tracking:
            ok, tracked_bbox = tracker.update(frame)
            if ok:
                    # x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    # cx = int(x + w // 2)
                    # cy = int(y + h // 2)
                    #cv2.line(frame, (int(cx), int(cy)), (width // 2, height), (255,255,255), 2)
                    
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
                    #cv2.putText(frame, text, (x,y-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
                follow_object(tracked_bbox, width)
            else :
                tracking = False
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
                if confidence > 0.4:
                    classes_score = row[5:]
                    ind = np.argmax(classes_score)
                    if classes_score[ind] > 0.4:
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
                        tracker = init_tracker(frame, bbox)
                        tracking = True
                        follow_object(bbox, width)
                        break
            # Display frame with bounding box
            #cv2.imshow("Frame", frame)

            #if cv2.waitKey(1) & 0xFF == ord('q'): 
                #break

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[tuple(state)])
    return action

def get_sensor_values(distances):
    k1 = 2
    k2 = 2
    k3 = 3
    k4 = 3

    # check the left side sensor
    dist = min(distances[0:3])
    if (dist > 40 and dist < 70):
        k1 = 1  #zone 1
    elif (dist <= 40):
        k1 = 0  #zone 0

    # check the right side sensor
    dist = min(distances[2:])
    if (dist > 40 and dist < 70):
        k2 = 1  #zone 1
    elif (dist <= 40):
        k2 = 0  #zone 0

    detected = [distance < 100 for distance in distances]
    # the left sector of the vehicle
    if detected[0] and detected[2]:
        k3 = 0 # both subsectors have obstacles
    elif (detected[1] or detected[2]) and not detected[0]:
        k3 = 1 # inner left subsector
    elif (detected[0] or detected[1]) and not detected[2]:
        k3 = 2 # outter left subsector

    # the right sector of the vehicle
    if detected[2] and detected[4]:
        k4 = 0 # both subsectors have obstacles
    elif (detected[2] or detected[3]) and not detected[4]:
        k4 = 1 # inner right subsector
    elif (detected[3] or detected[4]) and not detected[2]:
        k4 = 2 # outter right subsector 

    return [k1, k2, k3, k4]

Ab = AlphaBot()

cap = cv2.VideoCapture(0)
cam_cleaner = CameraBufferCleanerThread(cap)
net = cv2.dnn.readNetFromONNX("best.onnx")
        
classes = ['obstacle']

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracking = False

with open('q_table.pkl', 'rb') as f:
    Qtable_rlcar = pickle.load(f)

try:
    p1 = threading.Thread(target=detect)
    p1.start()

    while True:
        if tracking:
            time.sleep(0.1)
            continue

        distances = [(int(dist) if dist < 100 else 100) if dist >= 0 else 0 for dist in Ab.SR04()]
        print(distances, end=" ")
        state = get_sensor_values(distances)

        action = greedy_policy(Qtable_rlcar, state)
        if action == 0:
            print("Forward")
            Ab.forward()
        elif action == 1:
            print("Left")
            Ab.left()
        elif action == 2:
            print("Right")
            Ab.right()
        else:
            Ab.stop()
 
finally:
    GPIO.cleanup()
