import ffmpeg
import cv2
import subprocess
import sys
import re
import time

def get_ip_addr():
    # Run ifconfig command to get network interface information
    result = subprocess.run(['ifconfig', 'enp4s0'], capture_output=True, text=True)

    # Check if ifconfig command was successful
    if result.returncode == 0:
        # Use regular expression to find the IP address
        ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
        if ip_match:
            return ip_match.group(1)
        else:
            sys.exit("IP address not found")
    else:
        sys.exit("Failed to run ifconfig command")

HOST = get_ip_addr()

cap = cv2.VideoCapture("rtsp://192.168.0.101:8554/video_stream")  # Adjust the index or filename according to your source

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
codec_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

# Input parameters
input_params = {
    'f': 'rawvideo',
    'pix_fmt': 'bgr24',
    's': '{}x{}'.format(640, 480),
    'r': str(fps),
}

# FFmpeg output options
output_params = {
    'pix_fmt': 'yuvj422p',
    's': '{}x{}'.format(frame_width, frame_height),
    'r': str(fps),
    'f': 'rtsp',
    'vcodec': 'mjpeg',
}

# Input pipe
input_pipe = ffmpeg.input('pipe:', **input_params)

# Output RTSP stream
output_stream = ffmpeg.output(input_pipe, f'rtsp://{HOST}:8554/stream', **output_params)

# OpenCV video capture

# Run ffmpeg command
process = ffmpeg.run_async(output_stream, overwrite_output=True, pipe_stdin=True)

# Read frames from OpenCV and write to ffmpeg pipe
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    process.stdin.write(frame.tobytes())

# Release resources
cap.release()
process.stdin.close()
process.wait()
