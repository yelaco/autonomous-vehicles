import RPi.GPIO as GPIO
from AlphaBot import AlphaBot
import pickle
import numpy as np
import threading
import socket
import re
import sys
import subprocess
import time

def get_ip_addr():
    # Run ifconfig command to get network interface information
    result = subprocess.run(['ifconfig', 'wlan0'], capture_output=True, text=True)

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

def tcp_conn():
    global data
    global connected
    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the address and port
        server_socket.bind((HOST, PORT))
        
        # Listen for incoming connections
        server_socket.listen()
        print("Server is listening...")
        
        # Accept connection
        conn, addr = server_socket.accept()
        with conn:
            print('Connected by', addr)
            
            while connected:
                # Receive data from the client
                d = conn.recv(1024)
                if not d:
                    break
                
                data = d.decode()
                if "None" not in data:
                    if "Go straight" in data:
                        Ab.forward()
                        print("Received: Go straight")
                    elif "Turn left" in data:
                        Ab.left()
                        print("Received: Turn left")
                    elif "Turn right" in data:
                        Ab.right()
                        print("Received: Turn right")
                    elif "Stop" in data:
                        Ab.stop()
                        print("Received: Stop")
                        
    connected = False
            
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

HOST = get_ip_addr() 
PORT = 65432

# Start the rtsp stream #
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
    f"rtsp://{HOST}:8554/video_stream"
]
rtsp_stream = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(4)
#-----------------------#

data = "None"
connected = True

with open('config_files/q_table.pkl', 'rb') as f:
    Qtable_rlcar = pickle.load(f)

try:
    tcp_conn_thread = threading.Thread(target=tcp_conn)
    tcp_conn_thread.start()
    
    while connected:
        distances = [(int(dist) if dist < 100 else 100) if dist >= 0 else 0 for dist in Ab.SR04()]
        if "None" in data:
            print(distances, end=" ")
            state = get_sensor_values(distances)

            action = greedy_policy(Qtable_rlcar, state)
            if action == 0:
                print("Go straight")
                Ab.forward()
            elif action == 1:
                print("Turn left")
                Ab.left()
            elif action == 2:
                print("Turn right")
                Ab.right()
        else:
            if distances[2] < 10:
                Ab.stop()
                connected = False
                print("Stopped")
 
finally:
    GPIO.cleanup()
    rtsp_stream.terminate()
