import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import ipaddress
from base_station import BaseStation

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
		show_mode_selection()
		show_video_frame()
	except Exception:
		print("Couldn't init base station")

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

def show_video_frame():
	ip_label.pack_forget()  # Hide IP address input frame
	video_label.pack()  # Show video frame and mode selection
	show_mode_selection()  # Show mode selection widget

def show_mode_selection():
	mode_label.pack(pady=(20, 5))
	mode_switch.pack(pady=5)

def hide_mode_selection():
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

hide_mode_selection()
show_frame()
root.mainloop()
