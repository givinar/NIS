import socket
import numpy as np
import functions
import torch
from collections import namedtuple

Request = namedtuple('Request', 'name length')


class Client:
    def __init__(self):
        self.host = socket.gethostname()
        self.port = 8888
        self.client_socket = socket.socket()
        self.num_points = 10
        self.BUFSIZE = 8192
        self.points_data = None             # {x, y, z, s1, s2, pdf}
        self.put = Request(b'PUT', 3)
        self.put_ok = Request(b'PUT OK', 6)
        self.data_ok = Request(b'DATA OK', 7)

        # Just for tests
        self.ndims = 2
        self.funcname = 'Gaussian'
        self.function: functions.Function = getattr(functions, self.funcname)(n=self.ndims)

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

    def __get_samples(self):
        print("----> Infer")
        points = np.random.random((self.num_points, 3)).astype(np.float32)  # x, y, z
        raw_data = bytearray()
        raw_data.extend("i".encode())
        raw_data.extend(points.tobytes())
        self.client_socket.send(raw_data)
        data = self.client_socket.recv(1024)
        print('Received from server: ' + data.decode())
        self.client_socket.send(self.put_ok.name)
        self.__receive_raw()
        self.client_socket.send(self.data_ok.name)

        np_data = np.frombuffer(self.raw_data, dtype=np.float32)        # s1, s2, pdf
        samples = np_data.reshape((self.num_points, 3))
        self.points_data = np.concatenate((points, samples), axis=1)
        print(self.points_data)

    def __send_radiance(self):
        print("----> Train")
        t_data = torch.tensor(self.points_data[:, [3, 4]])
        t_y = self.function(t_data)
        y = t_y.cpu().detach().numpy()
        print(y)
        self.points_data = np.concatenate((self.points_data, y.reshape([self.num_points, 1])), axis=1)
        raw_data = bytearray()
        raw_data.extend("t".encode())
        raw_data.extend(self.points_data.tobytes())
        self.client_socket.send(raw_data)

        answer = self.client_socket.recv(1024)
        print('Received from server: ' + answer.decode()) # Data OK


    def __processing(self):
        while True:
            self.client_socket.send(self.put.name)
            answer = self.client_socket.recv(1024)
            print('Received from server: ' + answer.decode())
            if answer == self.put_ok.name:
                self.__get_samples()

            self.client_socket.send(self.put.name)
            answer = self.client_socket.recv(1024)
            print('Received from server: ' + answer.decode())
            if answer == self.put_ok.name:
                self.__send_radiance()

    def run_client(self):
        self.__connect()
        self.__processing()
        self.__disconnect()


if __name__ == '__main__':
    client = Client()
    client.run_client()
