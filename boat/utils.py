import socket
import subprocess
import sys
import re
import threading

def get_ip_addr():
	# Run ifconfig command to get network interface information
	result = subprocess.run(['ifconfig', 'wlp0s20f3'], capture_output=True, text=True)

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
		
class TcpConnThread(threading.Thread):
	def __init__(self, host, port, name='tcp-conn-thread'):
		self.host = host 
		self.port = port
		self.data = 'None'
		self.connected = False
		super(TcpConnThread, self).__init__(name=name)
		self.start()

	def run(self):	
		# Create a socket object
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
			# Bind the socket to the address and port
			server_socket.bind((self.host, self.port))
			
			# Listen for incoming connections
			server_socket.listen()
			print("Server is listening...")
			
			# Accept connection
			conn, addr = server_socket.accept()
			with conn:
				self.connected = True
				print('Connected by', addr)
				
				while connected:
					# Receive data from the client
					d = conn.recv(1024)
					if not d:
						break
					
					self.data = d.decode()
							
		connected = False
