import cv2
import numpy as np
import time

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[6]

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
            
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

net = cv2.dnn.readNetFromONNX("yolov5s.onnx")
file = open("coco.txt","r")
classes = file.read().split('\n')
print(classes)
# classes = ['obstacle']

decision = ""

bbox = None
tracking = False
obj_label = ""

# Main loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    if tracking:
        ok, bbox = tracker.update(frame)
        # Draw bounding box
        if ok:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cx = int(x + w // 2)
            cy = int(y + h // 2)
            cv2.line(frame, (int(cx), int(cy)), (width // 2, height), (255,255,255), 2)

            # Calculate slope
            if cx < width // 2 - 50: 
                decision = "Turn left"
            elif cx > width // 2 + 50: 
                decision = "Turn right"
            else:  
                decision = "Go straight"
            
            text = obj_label + "{:.2f}".format(conf)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(frame, text, (x,y-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
            cv2.putText(frame, decision, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
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
            if confidence > 0.5:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.2:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx- w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    wv= int(w * x_scale)
                    hv = int(h * y_scale)
                    box = np.array([x1,y1,wv,hv])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
        
        for i in indices:
            x1,y1,w,h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + "{:.2f}".format(conf)
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(255,0,0),2)
            cv2.putText(frame, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
            cv2.putText(frame, decision, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            
        if len(boxes) > 0:
            bbox = boxes[0]
            tracking = True
            tracker = init_tracker(frame, bbox)
            obj_label = classes[classes_ids[0]]

    # Display frame with bounding box
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    # # Add delay if needed
    # time.sleep(0.1)  # Adjust as needed based on the required frequency of sensor readings and decision making

# Release the camera and close serial connection
cap.release()
