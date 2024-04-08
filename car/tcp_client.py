import socket

# Define host and port
HOST = '192.168.0.100'
PORT = 65432

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # Connect to server
    client_socket.connect((HOST, PORT))
    
    # Send message to server
    while True:
        message = input("Enter your message: ")
        client_socket.sendall(message.encode())
