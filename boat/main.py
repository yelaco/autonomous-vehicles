import RPi.GPIO as GPIO
import pickle
import traceback
import threading
from boat import Airboat
from ultrasonic import Ultrasonic
from pid import PIDController
from utils import get_ip_addr, TcpConnThread
from obstacle_avoiding import greedy_policy, get_sensor_values
from video_streamer import stream_webcam

def measure(): 
    while not usv.is_shutdown:
        try:
            ultrasonic.measure_distances()
            tcp_conn_thread.send_data(f"{ultrasonic.latest_measure};")
        except Exception:
            traceback.print_exc()

def manual(data):
    global CONTROL_MODE
    if 'Manual: Left' in data:
        usv.left()
    elif 'Manual: Right' in data:
        usv.right()
    elif 'Manual: Forward' in data:
        usv.forward()
    elif 'Manual: Stop' in data:
        usv.stop()
    else:
        CONTROL_MODE = 2

def auto(data): 
    # action to avoid obstacle
    distances = ultrasonic.latest_measure
    oa_action = greedy_policy(Qtable_rlcar, get_sensor_values(distances))
    if oa_action ==	0: 
         # The vehicle keeps going straight as there is no osbstacle
        # Now we can continue detect and track bottles
        if 'Tracking' in data:
            # measured_value = measure()
            # usv.move(pid.update(measured_value))
            # ==> Plan to use pid controller later
            
            # temporary using old method
            if 'Tracking: Left' in data:
                usv.left()
            elif 'Tracking: Right' in data:
                usv.right()
            elif 'Tracking: Forward' in data:
                usv.forward()
        else:
            usv.forward()
    elif oa_action == 1:
        usv.left()
    elif oa_action == 2:
        usv.right()
    else:
        usv.stop()	

usv = Airboat()
HOST = get_ip_addr() 
PORT = 65432
rtsp_stream = stream_webcam(HOST)

ultrasonic = Ultrasonic(usv)
pid = PIDController(kp=0.5, ki=0.1, kd=0.2)

# Mode for controlling the boat
CONTROL_MODE = 2 # 0: shutdown, 1: manual, 2: auto

with open('config/q_table.pkl', 'rb') as f:
    Qtable_rlcar = pickle.load(f)

try:
    tcp_conn_thread = TcpConnThread(HOST, PORT)
    measure_thread = threading.Thread(target=measure)
    
    while tcp_conn_thread.running:
        if tcp_conn_thread.connected:
            # check for current control mode
            if 'Manual mode' in tcp_conn_thread.data:
                CONTROL_MODE = 1
            elif 'Auto mode' in tcp_conn_thread.data:
                CONTROL_MODE = 2
            elif 'Shutdown' in tcp_conn_thread.data:
                break
            else:
                # take action based on control mode
                if CONTROL_MODE == 1:
                    manual(tcp_conn_thread.data)
                elif CONTROL_MODE == 2: 
                    auto(tcp_conn_thread.data)
        else:
            auto('')
finally:
    print("Shutting down")
    ultrasonic.running = False
    usv.shutdown()
    rtsp_stream.terminate()
