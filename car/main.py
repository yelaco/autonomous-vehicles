import GPIO
import pickle
import threading
import traceback
from car import Car
from ultrasonic import Ultrasonic
from pid import PIDController
from utils import get_ip_addr, TcpConnThread
from obstacle_avoiding import greedy_policy, get_sensor_values
from video_streamer import stream_webcam

def measure(): 
    while not car.is_shutdown:
        try:
            ultrasonic.measure_distances()
            tcp_conn_thread.send_data(f"{ultrasonic.latest_measure};")
        except Exception:
            traceback.print_exc()

def manual(data):
    global CONTROL_MODE
    if 'Manual: Left' in data:
        car.left()
    elif 'Manual: Right' in data:
        car.right()
    elif 'Manual: Forward' in data:
        car.forward()
    elif 'Manual: Stop' in data:
        car.stop()
    else:
        CONTROL_MODE = 2

def auto(data): 
    # action to avoid obstacle
    distances = ultrasonic.latest_measure
    oa_action = greedy_policy(Qtable_rlcar, get_sensor_values(distances))
    if 'Tracking' in data:
            # measured_value = measure()
            # usv.move(pid.update(measured_value))
            # ==> Plan to use pid controller later

            if distances[2] < 5:
                car.stop()
                tcp_conn_thread.running = False
                tcp_conn_thread.connected = False
                return
            
            # temporary using old method
            if 'Tracking: Left' in data:
                car.left()
            elif 'Tracking: Right' in data:
                car.right()
            elif 'Tracking: Forward' in data:
                car.forward()
    else: 
        if all(d < 10 for d in distances):
            car.stop()
        elif oa_action == 0: 
            car.forward()
        elif oa_action == 1:
            car.left()
        elif oa_action == 2:
            car.right()
        else:
            car.stop()	

car = Car()
HOST = get_ip_addr() 
PORT = 65432
rtsp_stream = stream_webcam(HOST)

ultrasonic = Ultrasonic(car)
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
    car.shutdown()
    rtsp_stream.terminate()
