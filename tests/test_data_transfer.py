from multiprocessing import Process

import numpy as np

from client import Client
from integration_experiment import TrainServer, ExperimentConfig

np.random.seed(42)
TEST_DATA = np.random.rand(1000, 1000).astype(np.float32)


def test_data_transfer():
    client_process = Process(target=client_process_func)
    server_process = Process(target=server_process_func)
    server_process.start()
    client_process.start()
    client_process.join()
    server_process.join()


def server_process_func():
    server = TrainServer(ExperimentConfig())
    server.connect()

    # receive data
    server.receive_length()
    server.receive_raw()
    from_client = np.frombuffer(server.raw_data, dtype=np.float32).reshape(TEST_DATA.shape)
    assert np.array_equal(from_client, TEST_DATA), "Data from client differs"

    # send data
    raw_data = bytearray()
    raw_data.extend(TEST_DATA.tobytes())
    server.connection.send(len(TEST_DATA.tobytes()).to_bytes(4, 'little'))
    server.connection.sendall(raw_data)


def client_process_func():
    client = Client()
    client.connect()

    # send data
    client.client_socket.send(len(TEST_DATA.tobytes()).to_bytes(4, 'little'))  # bytes
    raw_data = bytearray()
    raw_data.extend(TEST_DATA.tobytes())
    client.client_socket.send(raw_data)

    # receive data
    client.receive_length()
    client.receive_raw()
    from_server = np.frombuffer(client.raw_data, dtype=np.float32).reshape(TEST_DATA.shape)
    assert np.array_equal(from_server, TEST_DATA), "Data from client differs"


def init_client_server():
    server = TrainServer(ExperimentConfig())
    client = Client()
    return client, server

