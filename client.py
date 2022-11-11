import socket
import numpy as np

class Client:
    def __init__(self):
        self.host = socket.gethostname()
        self.port = 8888
        self.client_socket = socket.socket()
        self.num_points = 10

    def __connect(self):
        print("Client: connecting..")
        self.client_socket.connect((self.host, self.port))
        print("Client: connected to the server")

    def __disconnect(self):
        self.client_socket.close()
        print("Client: disconnected from the server")

    def __inference(self):
        points = np.random.random((3, self.num_points))
        message = bytearray()
        message.extend("i".encode())
        message.extend(points.tobytes())
        self.client_socket.send(message)

    def __processing(self):
        while True:
            message = "PUT"
            self.client_socket.send(message.encode())
            data = self.client_socket.recv(1024).decode()
            print('Received from server: ' + data)
            #IF OK
            self.__inference()



    def run_client(self):

        self.__connect()
        self.__processing()
        self.__disconnect()