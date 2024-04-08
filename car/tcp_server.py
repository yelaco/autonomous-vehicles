import socket

# Define host and port
HOST = '192.168.0.100'
PORT = 65432

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
        while True:
            # Receive data from the client
            data = conn.recv(1024)
            if not data:
                break
            print("Received:", data.decode())