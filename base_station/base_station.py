import numpy as np
from video_proc import VideoProcessor
from utils import tcp_client

class BaseStation:
	def __init__(self, mode='detect'):
		self.connected = False
		self.mode = mode
	
	def connect(self, host, port=65432):
		self.send_command, self.close_connection = tcp_client(host, port)
		self.vproc = VideoProcessor(self.host)
	
	def get_decision(bbox, width=640):
		x, w = int(bbox[0]), int(bbox[2])
		cx = int(x + w // 2)

		# Calculate slope
		if cx < width // 2 - 50: 
			return "Tracking: Left"
		elif cx > width // 2 + 50: 
			return "Tracking: Right"
		else:  
			return "Tracking: Forward"
	
	def real_time_control(self, command='None'):
		if self.connected:
			frame = self.vproc.get_latest_frame()

			if self.mode == 'tracking':
				ok, bbox = self.vproc.tracking(frame)
				if ok:
					self.send_command(self.get_decision(bbox, frame.shape[1]))
				else :
					self.mode = 'detect'
			elif self.mode == 'detect': 
				detected, bbox = self.vproc.detect(frame)
				if detected:	
					self.vproc.tracker = self.vproc.init_tracker(frame, bbox)
					self.mode = 'tracking'
				else:
					self.send_command('None')
			elif self.mode == 'manual':
				self.send_command(command)

			return True, frame
		else:
			return False, None 

	def close(self):
		self.close_connection()
