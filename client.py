import socket
import numpy as np
import functions
import torch
from collections import namedtuple

import logging
import sys


Request = namedtuple('Request', 'name length')

HOST = "127.0.0.1"
PORT = 65432

class Client:
    def __init__(self):
        self.host = HOST
        self.port = PORT
        self.client_socket = socket.socket()
        self.num_points = 10
        self.BUFSIZE = 8192
        self.points_data = None             # {x, y, z, x_norm, y_norm, z_norm, s1, s2, s3 pdf}
        self.put_infer = Request(b'PUT INFER', 9)
        self.put_train = Request(b'PUT TRAIN', 9)
        self.put_infer_ok = Request(b'PUT INFER OK', 12)
        self.put_train_ok = Request(b'PUT TRAIN OK', 12)
        self.data_ok = Request(b'DATA OK', 7)
        self.length = 0

        # Just for tests
        self.ndims = 3
        self.funcname = 'Gaussian'
        self.function: functions.Function = getattr(functions, self.funcname)(n=self.ndims)

    def receive_raw(self):
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

    def connect(self):
        print("Client: connecting..")
        self.client_socket.connect((self.host, self.port))
        print("Client: connected to the server")

    def __disconnect(self):
        self.client_socket.close()
        print("Client: disconnected from the server")

    def get_samples(self):
        print("----> Infer")
        points = np.random.random((self.num_points, 9)).astype(np.float32)  # pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, dir_x, dir_y, dir_z
                                                                            # dir_? right now is not taken into account right now
        self.client_socket.send(len(points[:, [0, 1, 2, 3, 4, 5]].tobytes()).to_bytes(4, 'little'))  # bytes

        raw_data = bytearray()
        raw_data.extend(points[:, [0, 1, 2, 3, 4, 5]].tobytes())
        self.client_socket.send(raw_data)

        data = self.client_socket.recv(self.put_infer.length)
        print('Received from server: ' + data.decode())
        if data == self.put_infer.name:
            self.client_socket.send(self.put_infer_ok.name)
            self.receive_length()
            self.receive_raw()
            self.client_socket.send(self.data_ok.name)

        np_data = np.frombuffer(self.raw_data, dtype=np.float32)        # s1, s2, s3, pdf
        samples = np_data.reshape((self.num_points, -1))
        self.points_data = np.concatenate((points, samples), axis=1)

    def send_radiance(self):
        print("----> Train")
        t_data = torch.tensor(self.points_data[:, [9, 10, 11]]) # s1, s2, s3
        t_y = self.function(t_data)
        y = t_y.cpu().detach().numpy()
        lum = np.stack((y,) * 3, axis=-1)
        scale = np.array([3, 3, 3])
        lum = lum / scale
        self.points_data = np.concatenate((self.points_data, lum.reshape([len(lum), 3])), axis=1, dtype=np.float32)
        self.client_socket.send(len(self.points_data[:, [13,14,15,0,1,2,3,4,5,6,7,8]].tobytes()).to_bytes(4, 'little'))  # bytes

        raw_data = bytearray()
        raw_data.extend(self.points_data[:, [13,14,15,0,1,2,3,4,5,6,7,8]].tobytes())    #send r,g,b x, y, z, norm_x, norm_y, norm_z, dir_x, dir_y, dir_z
        self.client_socket.send(raw_data)

        answer = self.client_socket.recv(self.data_ok.length)
        print('Received from server: ' + answer.decode()) # Data OK


    def __processing(self):
        while True:
            self.client_socket.send(self.put_infer.name)
            answer = self.client_socket.recv(self.put_infer_ok.length)
            print('Received from server: ' + answer.decode())
            if answer == self.put_infer_ok.name:
                self.get_samples()

            self.client_socket.send(self.put_train.name)
            answer = self.client_socket.recv(self.put_train_ok.length)
            print('Received from server: ' + answer.decode())
            if answer == self.put_train_ok.name:
                self.send_radiance()

    def run_client(self):
        self.connect()
        self.__processing()
        self.__disconnect()


if __name__ == '__main__':
    client = Client()
    client.run_client()
