import socket
import RPi.GPIO as GPIO
from AlphaBot import AlphaBot
import pickle
import numpy as np

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

# Define host and port
HOST = '192.168.0.100'
PORT = 65432

with open('q_table.pkl', 'rb') as f:
    Qtable_rlcar = pickle.load(f)
    
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
        
        conn.settimeout(0.1)
        while True:
            try:
                # Receive data from the client
                data = conn.recv(1024)
                if not data:
                    break
                    
                print("Received:", data.decode())
            except socket.timeout:
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
                break