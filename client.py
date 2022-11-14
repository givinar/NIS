import socket
import numpy as np
from collections import namedtuple

Request = namedtuple('Request', 'name length')


class Client:
    def __init__(self):
        self.host = socket.gethostname()
        self.port = 8888
        self.client_socket = socket.socket()
        self.num_points = 10
        self.BUFSIZE = 8192
        self.put = Request(b'PUT', 3)
        self.put_ok = Request(b'PUT OK', 6)
        self.data_ok = Request(b'DATA OK', 7)

    def __receive_raw(self):
        try:
            self.raw_data = bytearray()
            while True:
                    chunk = self.client_socket.recv(self.BUFSIZE)
                    self.raw_data.extend(chunk)
                    if len(chunk) < self.BUFSIZE:
                        break
        except ConnectionError:
            print(f"Error: Connection failed ...\n")

    def __connect(self):
        print("Client: connecting..")
        self.client_socket.connect((self.host, self.port))
        print("Client: connected to the server")

    def __disconnect(self):
        self.client_socket.close()
        print("Client: disconnected from the server")

    def __inference(self):
        print("----> Infer")
        points = np.random.random((self.num_points, 3))
        raw_data = bytearray()
        raw_data.extend("i".encode())
        raw_data.extend(points.tobytes())
        self.client_socket.send(raw_data)
        data = self.client_socket.recv(1024)
        print('Received from server: ' + data.decode())
        self.client_socket.send(self.put_ok.name)
        self.__receive_raw()
        self.client_socket.send(self.data_ok.name)

        np_data = np.frombuffer(self.raw_data, dtype=np.float32)
        print(np_data)

    def __processing(self):
        while True:
            self.client_socket.send(self.put.name)
            answer = self.client_socket.recv(1024)
            print('Received from server: ' + answer.decode())
            if answer == self.put_ok.name:
                self.__inference()

    def run_client(self):
        self.__connect()
        self.__processing()
        self.__disconnect()


if __name__ == '__main__':
    client = Client()
    client.run_client()
