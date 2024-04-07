import RPi.GPIO as GPIO
 
import time
import sys

#from mpu6050 import mpu6050
 
 
class AlphaBot(object):
  
 
    def __init__(self,in1=13,in2=12,ena=6,in3=21,in4=20,enb=26, trig_1=17, echo_1=18, trig_2=23, echo_2=24, trig_3=27, echo_3=22,trig_4=4,echo_4=25,trig_5=16,echo_5=19):
 
        self.IN1 = in1
 
        self.IN2 = in2
 
        self.IN3 = in3
 
        self.IN4 = in4
 
        self.ENA = ena
 
        self.ENB = enb

        self.TRIG_1 = trig_1
        self.ECHO_1 = echo_1
        self.TRIG_2 = trig_2
        self.ECHO_2 = echo_2
        self.TRIG_3 = trig_3
        self.ECHO_3 = echo_3
        #Add 2 sensor
        self.TRIG_4 = trig_4
        self.ECHO_4 = echo_4
        self.TRIG_5 = trig_5
        self.ECHO_5 = echo_5

        GPIO.setmode(GPIO.BCM)
 
        GPIO.setwarnings(False)
 
        GPIO.setup(self.IN1,GPIO.OUT)
 
        GPIO.setup(self.IN2,GPIO.OUT)
 
        GPIO.setup(self.IN3,GPIO.OUT)
 
        GPIO.setup(self.IN4,GPIO.OUT)
 
        GPIO.setup(self.ENA,GPIO.OUT)
 
        GPIO.setup(self.ENB,GPIO.OUT)
        
        # set up for HC-SR04-1
        GPIO.setup(self.ECHO_1, GPIO.IN)
        GPIO.setup(self.TRIG_1, GPIO.OUT)
        # set up for HC-SR04-2
        GPIO.setup(self.ECHO_2, GPIO.IN)
        GPIO.setup(self.TRIG_2, GPIO.OUT)
        # set up for HC-SR04-3
        GPIO.setup(self.ECHO_3, GPIO.IN)
        GPIO.setup(self.TRIG_3, GPIO.OUT)  
        # set up for HC-SR04-4
        GPIO.setup(self.ECHO_4, GPIO.IN)
        GPIO.setup(self.TRIG_4, GPIO.OUT)
        # set up for HC-SR04-5
        GPIO.setup(self.ECHO_5, GPIO.IN)
        GPIO.setup(self.TRIG_5, GPIO.OUT) 
        # set up for MPU-6050
        #GPIO.setup(self.SCL, GPIO.IN)
        #GPIO.setup(self.SDA, GPIO.IN)
 
        self.stop()
 
        self.PWMA = GPIO.PWM(self.ENA,500)
 
        self.PWMB = GPIO.PWM(self.ENB,500)
 
        self.PWMA.start(40)
 
        self.PWMB.start(40)
 
    def SR04(self):
        trigs=  [self.TRIG_3, self.TRIG_4, self.TRIG_2, self.TRIG_5, self.TRIG_1]
        echos= [self.ECHO_3, self.ECHO_4, self.ECHO_2, self.ECHO_5, self.ECHO_1]
        distances = [100, 100, 100, 100, 100]
        for i in range(5):
            #print("i:::",i) 
            trig = trigs[i]
            echo = echos[i]
            GPIO.output(trig, GPIO.LOW)
            time.sleep(0.02)
            GPIO.output(trig, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(trig, GPIO.LOW)

            start = time.time()
            pulse_start = start
            while GPIO.input(echo) == 0 and pulse_start - start < 0.005:
                pulse_start = time.time()
                if round(pulse_start - start, 3) >= 0.300:
                    sys.exit(f"HR-SO4 #{i+1} timed out")
            else:
                pulse_end = 0
                pulse_duration = 0
                while GPIO.input(echo) == 1 and pulse_duration < 0.1:
                    pulse_end = time.time()
                    pulse_duration = pulse_end - pulse_start

            pulse_duration = pulse_end - pulse_start

            # v (cm/s)
            distance = pulse_duration * 17150
            distances[i] = distance - 2 

        return distances

    def MPU(self):
        time.sleep(0.2)
        mpu = mpu6050(0x68)
        accel_data = mpu.get_accel_data()
        gyro_data = mpu.get_gyro_data()
        time.sleep(1)
        return [accel_data, gyro_data]
        

    def forward(self):
        self.PWMA.start(60)
        self.PWMB.start(60)
 
        GPIO.output(self.IN1,GPIO.LOW) # Tien_phai
 
        GPIO.output(self.IN2,GPIO.HIGH) # Lui_phai
 
        GPIO.output(self.IN3,GPIO.LOW) # Tien_trai
 
        GPIO.output(self.IN4,GPIO.HIGH) # Lui_trai
 
 
 
    def stop(self):
 
        GPIO.output(self.IN1,GPIO.LOW)
 
        GPIO.output(self.IN2,GPIO.LOW)
 
        GPIO.output(self.IN3,GPIO.LOW)
 
        GPIO.output(self.IN4,GPIO.LOW)
 
 
 
    def backward(self):
 
        GPIO.output(self.IN1,GPIO.HIGH)
 
        GPIO.output(self.IN2,GPIO.LOW)
 
        GPIO.output(self.IN3,GPIO.HIGH)
 
        GPIO.output(self.IN4,GPIO.LOW)
 
 
 
    def right(self):

        self.PWMA.start(80)
        self.PWMB.start(80)
 
        GPIO.output(self.IN1,GPIO.LOW)
 
        GPIO.output(self.IN2,GPIO.HIGH)
 
        GPIO.output(self.IN3,GPIO.LOW)
 
        GPIO.output(self.IN4,GPIO.LOW)
 
 
 
    def left(self):
 
        self.PWMA.start(80)
        self.PWMB.start(80)

        GPIO.output(self.IN1,GPIO.LOW)
 
        GPIO.output(self.IN2,GPIO.LOW)
 
        GPIO.output(self.IN3,GPIO.LOW)
 
        GPIO.output(self.IN4,GPIO.HIGH)
 
       
 
    def setPWMA(self,value):
 
        self.PWMA.ChangeDutyCycle(value)
 
 
 
    def setPWMB(self,value):
 
        self.PWMB.ChangeDutyCycle(value)   
 
       
 
    def setMotor(self, left, right):
 
        if((right >= 0) and (right <= 100)):
 
            GPIO.output(self.IN1,GPIO.HIGH)
 
            GPIO.output(self.IN2,GPIO.LOW)
 
            self.PWMA.ChangeDutyCycle(right)
 
        elif((right < 0) and (right >= -100)):
 
            GPIO.output(self.IN1,GPIO.LOW)
 
            GPIO.output(self.IN2,GPIO.HIGH)
 
            self.PWMA.ChangeDutyCycle(0 - right)
 
        if((left >= 0) and (left <= 100)):
 
            GPIO.output(self.IN3,GPIO.HIGH)
 
            GPIO.output(self.IN4,GPIO.LOW)
 
            self.PWMB.ChangeDutyCycle(left)
 
        elif((left < 0) and (left >= -100)):
 
            GPIO.output(self.IN3,GPIO.LOW)
 
            GPIO.output(self.IN4,GPIO.HIGH)
 
            self.PWMB.ChangeDutyCycle(0 - left)
