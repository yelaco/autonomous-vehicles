import socket
import traceback
import subprocess
import time
import sys
import random

host = '127.0.0.1'
port = 65432
connected = True
running = True 

try:
    mediamtx_pid = subprocess.check_output(["pidof", "mediamtx"]).decode().strip()
except:
    sys.exit("RTSP server hasn't been started. Read HOW_TO_RUN.txt in /car folder")

ffmpeg_command = [
    "ffmpeg",
    "-f", "v4l2",
    "-framerate", "60",
    "-re",
    "-stream_loop", "-1",
    "-video_size", "640x480",
    "-input_format", "mjpeg",
    "-i", "/dev/video0",
    "-c", "copy",
    "-f", "rtsp",
    f"rtsp://{host}:8554/video_stream"
]
rtsp_stream = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(4)

def send_data(data):
    """Send data back to the connected client."""
    if connected:
        try:
            conn.sendall(data.encode())
        except Exception as e:
            print(f"Error sending data: {e}")

def test_send_data():
    random_array = [random.randint(0, 100) for _ in range(5)]
    send_data(f"{random_array}")
                
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    # Bind the socket to the address and port
    server_socket.bind((host, port))
            
    # Listen for incoming connections
    server_socket.listen()
    print("Server is listening...")
        
    while running:
        conn, addr = server_socket.accept()
        try:
            with conn:
                connected = True
                print('Connected by', addr)
                        
                while connected:
                    # Receive data from the client
                    d = conn.recv(1024)
                    if not d:
                        break
                    
                    data = d.decode()

                    print(data)

                    if "Which" in data:
                        send_data("Vehicle: Car")
                        print("HERE")

                    if "Disconnect" in data:
                        connected = False
                    elif "Shutdown" in data:
                        connected = False
                        running = False
                    
                    test_send_data()

        except Exception:
            traceback.print_exc()
        connected = False
