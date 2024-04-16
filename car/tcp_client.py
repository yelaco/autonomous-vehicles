import socket
import time
import cv2
import numpy as np
import threading
import argparse
import ipaddress
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[5]

parser = argparse.ArgumentParser(description='Connect to raspberry pi using IP address')
parser.add_argument('host', metavar='h', type=str, help='ip address of pi')
args = parser.parse_args()

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
        self.running = True
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while self.running:
            _, self.last_frame = self.camera.read()
    
def get_decision(bbox):
    x, w = int(bbox[0]), int(bbox[2])
    cx = int(x + w // 2)

    # Calculate slope
    if cx < width // 2 - 50: 
        return "Turn left"
    elif cx > width // 2 + 50: 
        return "Turn right"
    else:  
        return "Go straight"

def count_red_pixels(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v_eq])
    
    # Define lower and upper bounds for red color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Mask out red pixels
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Define lower and upper bounds for slightly different red color
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Mask out the second range of red pixels
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Count red pixels
    red_pixel_count = cv2.countNonZero(mask)

    return red_pixel_count

# Define host and port
try:
    HOST = str(ipaddress.ip_address(args.host))
    print(f"Valid IP address: {HOST}")
except ValueError:
    sys.exit(f"Invalid IP address: {args.host}")
PORT = 65432

cap = cv2.VideoCapture(f"rtsp://{HOST}:8554/video_stream")
cam_cleaner = CameraBufferCleanerThread(cap)

net = cv2.dnn.readNetFromONNX("best1404.onnx")
classes = ['obstacle', 'parking']

bbox = None
tracking = False
obj_label = ""
conf = -1

decision = ""

threshold = 120000

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # Connect to server
    client_socket.connect((HOST, PORT))
    
    # Main loop
    while True and decision != "Stop":
        # Read frame from webcam
        if cam_cleaner.last_frame is None:
            continue

        frame = cam_cleaner.last_frame

        # Start timer
        timer = cv2.getTickCount()

        if tracking:
            ok, bbox = tracker.update(frame)
            # Draw bounding box
            if ok:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cx = int(x + w // 2)
                cy = int(y + h // 2)
                cv2.line(frame, (int(cx), int(cy)), (width // 2, height), (255,255,255), 2)

                decision = get_decision(bbox) 
                
                num_red_pixel = count_red_pixels(frame)
                if num_red_pixel > threshold:
                    decision = "Stop"
                    tracking = False
                else:
                    print(num_red_pixel)
                    
                client_socket.sendall(decision.encode())
                text = obj_label + "{:.2f}".format(conf)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
                cv2.putText(frame, text, (x,y-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
                cv2.putText(frame, decision, (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)  
            else :
                tracking = False
        else: 
            # Detect objects using YOLOv5
            height, width, _ = frame.shape
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
                if confidence >= 0.4:
                    classes_score = row[5:]
                    ind = np.argmax(classes_score)
                    if classes_score[ind] >= 0.25:
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
                num_retained_boxes = cv2.dnn.NMSBoxes(boxes,confidences,0.25,0.45)
                parking_id = -1
                max_conf = -1
                for i in num_retained_boxes:
                    if classes[classes_ids[i]] == 'parking' and max_conf < confidences[i]:
                        parking_id = i
                        max_conf = confidences[i]
                    
                if parking_id > -1:
                    bbox = boxes[parking_id]
                    x1,y1,w,h = bbox
                    obj_label = classes[classes_ids[parking_id]]
                    conf = max_conf
                    text = obj_label + "{:.2f}".format(conf)
                    cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(255,0,0),2)
                    cv2.putText(frame, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
                    cv2.putText(frame, decision, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    tracker = init_tracker(frame, bbox)
                    tracking = True
                    client_socket.sendall(get_decision(bbox).encode())
                        
            
            if not tracking:
                client_socket.sendall("None".encode())
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        
        # Display frame with bounding box
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
    cam_cleaner.running = False
    time.sleep(2)
    cap.release()
    client_socket.close()    
