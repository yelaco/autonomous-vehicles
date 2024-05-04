import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import ipaddress
from base_station import BaseStation
import logging
import traceback

def switch_bs_mode(*args):
	mode = mode_var.get()
	if mode == "Auto":	
		bs.mode = 'detect' 
		bs.send_command("Auto mode")
	elif mode == "Manual":
		bs.mode = 'manual'
		bs.send_command("Manual mode")

def on_connect():
	global bs
	try:
		ip_address = str(ipaddress.ip_address(ip_entry.get()))	
		print(f"Valid ip address: {ip_address}")
		bs.connect(ip_address)
		show_start_sreen(False)
		show_boat_webcam(True)
		show_mode_selection(True)
	except Exception as e:
		traceback.print_exc()
		print("Couldn't init base station")

def on_manual(event):
	if bs.mode == 'manual':
		if event.keysym == "Left":
			bs.send_command("Manual: Left")
		elif event.keysym == "Right":
			bs.send_command("Manual: Right")
		elif event.keysym == "Up":
			bs.send_command("Manual: Forward")
		elif event.keysym == "Down":
			bs.send_command("Manual: Stop")

# Init base station
bs = BaseStation()

root = tk.Tk()
root.title("Base Station v0.1")

# IP address input and Connect button
ip_label = tk.Label(root, text="Enter Boat Ip:")
ip_label.pack(pady=(10, 5))
ip_entry = tk.Entry(root)
ip_entry.pack(pady=5)
connect_button = tk.Button(root, text="Connect", command=on_connect)
connect_button.pack(pady=5)

# Video frame display
video_label = tk.Label(root)
video_label.pack()

# Mode selection (Auto/Manual)
mode_label = tk.Label(root, text="Mode:")
mode_label.pack(pady=(20, 5))
mode_var = tk.StringVar(value="Auto")
mode_var.trace_add('write', switch_bs_mode)
mode_switch = ttk.Combobox(root, textvariable=mode_var, values=["Auto", "Manual"], state="readonly")
mode_switch.pack(pady=5)

root.bind("<Left>", on_manual)
root.bind("<Right>", on_manual)
root.bind("<Up>", on_manual)
root.bind("<Down>", on_manual)

def show_start_sreen(is_displayed=True):
	if is_displayed:
		ip_label.pack(pady=(10, 5))
		ip_entry.pack(pady=5)
		connect_button.pack(pady=5)
	else:
		ip_label.pack_forget()
		ip_entry.pack_forget()
		connect_button.pack_forget()
 
def show_boat_webcam(is_displayed=True):
	if is_displayed:
		video_label.pack()  # Show video frame and mode selection
	else:
		video_label.pack_forget()

def show_mode_selection(is_displayed=True):
	if is_displayed:
		mode_label.pack(pady=(20, 5))
		mode_switch.pack(pady=5)
	else:
		mode_label.pack_forget()
		mode_switch.pack_forget()

def show_frame():
	ret, frame = bs.real_time_control()
	if ret:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(frame)
		img = ImageTk.PhotoImage(image=img)
		video_label.img = img
		video_label.config(image=img)
	video_label.after(50, show_frame)

show_mode_selection(False)
show_frame()
root.mainloop()
