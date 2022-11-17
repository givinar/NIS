import socket
import numpy as np
import functions
import torch
from collections import namedtuple

import logging
import sys


Request = namedtuple('Request', 'name length')


class Client:
    def __init__(self):
        self.host = socket.gethostname()
        self.port = 8888
        self.client_socket = socket.socket()
        self.num_points = 10
        self.BUFSIZE = 8192
        self.points_data = None             # {x, y, z, s1, s2, pdf}
        self.put_infer = Request(b'PUT INFER', 9)
        self.put_train = Request(b'PUT TRAIN', 9)
        self.put_infer_ok = Request(b'PUT INFER OK', 12)
        self.put_train_ok = Request(b'PUT TRAIN OK', 12)
        self.data_ok = Request(b'DATA OK', 7)
        self.length = 0

        # Just for tests
        self.ndims = 2
        self.funcname = 'Gaussian'
        self.function: functions.Function = getattr(functions, self.funcname)(n=self.ndims)

    def __receive_raw(self):
        self.raw_data = bytearray()
        bytes_recd = 0
        while bytes_recd < self.length:
            chunk = self.client_socket.recv(min(self.length - bytes_recd, self.BUFSIZE))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            self.raw_data.extend(chunk)
            bytes_recd = bytes_recd + len(chunk)

    def receive_length(self):
        self.length = int.from_bytes(self.client_socket.recv(4), 'little')

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
        self.client_socket.send(len(points.tobytes()).to_bytes(4, 'little'))  # bytes

        raw_data = bytearray()
        raw_data.extend(points.tobytes())
        self.client_socket.send(raw_data)

        data = self.client_socket.recv(self.put_infer.length)
        print('Received from server: ' + data.decode())
        if data == self.put_infer.name:
            self.client_socket.send(self.put_infer_ok.name)
            self.receive_length()
            self.__receive_raw()
            self.client_socket.send(self.data_ok.name)

        np_data = np.frombuffer(self.raw_data, dtype=np.float32)        # s1, s2, pdf
        samples = np_data.reshape((self.num_points, -1))
        self.points_data = np.concatenate((points, samples), axis=1)

    def __send_radiance(self):
        print("----> Train")
        t_data = torch.tensor(self.points_data[:, [3, 4]])
        t_y = self.function(t_data)
        y = t_y.cpu().detach().numpy()
        self.points_data = np.concatenate((self.points_data, y.reshape([self.num_points, 1])), axis=1)
        self.client_socket.send(len(self.points_data[:, [0, 1, 2, 6]].tobytes()).to_bytes(4, 'little'))  # bytes

        raw_data = bytearray()
        raw_data.extend(self.points_data[:, [0, 1, 2, 6]].tobytes())    #send x, y, z, rad
        self.client_socket.send(raw_data)

        answer = self.client_socket.recv(self.data_ok.length)
        print('Received from server: ' + answer.decode()) # Data OK


    def __processing(self):
        while True:
            self.client_socket.send(self.put_infer.name)
            answer = self.client_socket.recv(self.put_infer_ok.length)
            print('Received from server: ' + answer.decode())
            if answer == self.put_infer_ok.name:
                self.__get_samples()

            self.client_socket.send(self.put_train.name)
            answer = self.client_socket.recv(self.put_train_ok.length)
            print('Received from server: ' + answer.decode())
            if answer == self.put_train_ok.name:
                self.__send_radiance()

    def run_client(self):
        self.__connect()
        self.__processing()
        self.__disconnect()


if __name__ == '__main__':
    client = Client()
    client.run_client()
