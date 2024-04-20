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

# Input parameters
input_params = {
    'f': 'rawvideo',
    'pix_fmt': 'bgr24',
    's': '640x480',
}

# Output parameters
output_params = {
    "f": "rtsp",
}

# Input pipe
input_pipe = ffmpeg.input('pipe:', **input_params)

# Output RTSP stream
output_stream = ffmpeg.output(input_pipe, f'rtsp://{HOST}:8554/stream', **output_params)

# OpenCV video capture
cap = cv2.VideoCapture("test.mp4")  # Adjust the index or filename according to your source

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
