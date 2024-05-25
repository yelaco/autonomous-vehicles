# import the opencv library 
import cv2 
import subprocess
import inference
import numpy as np
  
def open_ffmpeg_stream_process():
    command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(640, 480),
           '-r', "30",
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           "rtmp://35.240.198.118:1935/live/stream"]
    return subprocess.Popen(command, stdin=subprocess.PIPE)

model = inference.get_model(
    model_id="water-waste-eouzy/2", # Roboflow model to use
    api_key="hCR4jqAHhoEQhXUytxsJ"
)

ffmpeg_process = open_ffmpeg_stream_process()

# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    if not ret: break

    # Inference image to find faces
    results = model.infer(image=frame, confidence=0.5, iou_threshold=0.5)

    # Plot image with face bounding box (using opencv)
    if results[0].predictions:
        prediction = results[0].predictions[0]

        x, y, w, h = int(prediction.x), int(prediction.y), int(prediction.width), int(prediction.height)
            
        cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (255,255,0), 5)
        cv2.putText(frame, prediction.class_name, (x - 10, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{prediction.confidence:.2f}", (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the resulting frame 
    cv2.imshow('frame', frame)
    
    ffmpeg_process.stdin.write(frame.tobytes()) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
# Destroy all the windows 
cv2.destroyAllWindows() 
