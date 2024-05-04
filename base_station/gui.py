import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import ipaddress
from base_station import BaseStation
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
		bs.connect(host=ip_address, in_msg_label=in_msg_label, out_msg_label=out_msg_label)
		show_start_sreen(False)
		show_work_screen()
	except Exception:
		traceback.print_exc()
		print("Couldn't init base station")

def on_disconnect():
	if messagebox.askokcancel("Disconnect", "Do you want to disconnect and close the application?"):
		bs.send_command("Disconnect")
		bs.close()
		root.destroy()

def on_manual(event):
	if bs.mode == 'manual':
		if event.keysym == "Left":
			bs.send_command("Manual: Left")
		elif event.keysym == "Right":
			bs.send_command("Manual: Right")
		elif event.keysym == "Up":
			bs.send_command("Manual: Forward")
		elif event.keysym == 's':
			bs.send_command("Manual: Stop")

def on_shutdown():
	if messagebox.askokcancel("Shutdown", "Do you want to shutdown the boat and close the application?"):
		bs.send_command("Shutdown")
		bs.close()
		root.destroy()

# Init base station
bs = BaseStation()

root = tk.Tk()
root.title("Base Station v0.1")
root.geometry("640x600")

canvas = tk.Canvas(root, width=640, height=600)

work_frame = tk.Frame(root)

# IP address input and Connect button
ip_label = tk.Label(canvas, text="Enter Boat Ip:")
ip_entry = tk.Entry(canvas)
connect_button = tk.Button(canvas, text="Connect", command=on_connect)

# Create wallpaper label
bg_image = Image.open("config/wallpaper.png") 
bg_image = bg_image.resize((640, 600))
bg_image = ImageTk.PhotoImage(bg_image)
canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)

canvas.create_window(240, 80, anchor=tk.NW, window=ip_label)
canvas.create_window(240, 110, anchor=tk.NW, window=ip_entry)
canvas.create_window(240, 150, anchor=tk.NW, window=connect_button)

# Create logo label
logo_image = Image.open("config/logo_uet.png") 
logo_image = logo_image.resize((100, 100))
logo_image = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(work_frame, image=logo_image)

# Video frame display
video_label = tk.Label(work_frame)
video_label.grid(row=0, columnspan=3)

# Mode selection (Auto/Manual)
mode_label = tk.Label(work_frame, text="Mode")
mode_var = tk.StringVar(value="Auto")
mode_var.trace_add('write', switch_bs_mode)
mode_switch = ttk.Combobox(work_frame, textvariable=mode_var, values=["Auto", "Manual"], state="readonly")

out_msg_label = tk.Label(work_frame, text="Sent:")
in_msg_label = tk.Label(work_frame, text="Received:")

shutdown_button = tk.Button(work_frame, text="Shutdown", command=on_shutdown)
disconnect_button = tk.Button(work_frame, text="Disconnect", command=on_disconnect)

root.bind("<Left>", on_manual)
root.bind("<Right>", on_manual)
root.bind("<Up>", on_manual)
root.bind("s", on_manual)

def show_start_sreen(is_displayed=True):
	if is_displayed:
		canvas.pack(fill=tk.BOTH, expand=True)
	else:
		canvas.pack_forget()

def show_work_screen(is_displayed=True):
	if is_displayed:
		show_logo()
		show_boat_webcam()
		show_mode_selection()
		show_messages()
		shutdown_button.grid(row=3, column=3, padx=10, pady=10, sticky=tk.SE)
		disconnect_button.grid(row=3, column=2, pady=10, sticky=tk.SE)
		work_frame.pack(fill=tk.BOTH, expand=True)
	else:
		show_logo(False)
		show_boat_webcam(False)
		show_mode_selection(False)
		show_messages(False)
		shutdown_button.grid_forget()
		disconnect_button.grid_forget()
		work_frame.pack_forget()
  
def show_logo(is_displayed=True):
	if is_displayed:
		logo_label.grid(row=1, column=0, rowspan=3, padx=5, pady=10, sticky=tk.W) 
	else:
		logo_label.grid_forget()
 
def show_boat_webcam(is_displayed=True):
	if is_displayed:
		video_label.grid(row=0, column=0, columnspan=4)  # Show video frame and mode selection
	else:
		video_label.grid_forget()

def show_mode_selection(is_displayed=True):
	if is_displayed:
		mode_label.grid(row=1, column=1, sticky=tk.SW)
		mode_switch.grid(row=2, column=1, sticky=tk.NW)
	else:
		mode_label.grid_forget()
		mode_switch.grid_forget()
	
def show_messages(is_displayed=True):
	if is_displayed:
		out_msg_label.grid(row=1, column=2, columnspan=2, sticky=tk.SW)
		in_msg_label.grid(row=2, column=2, columnspan=2, sticky=tk.NW)
	else:
		in_msg_label.grid_forget()
		out_msg_label.grid_forget()

def show_frame():
	ret, frame = bs.real_time_control()
	if ret:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(frame)
		img = ImageTk.PhotoImage(image=img)
		video_label.img = img
		video_label.config(image=img)
	video_label.after(50, show_frame)

show_start_sreen()
show_work_screen(False)
show_frame()
root.mainloop()
