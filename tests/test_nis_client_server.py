from multiprocessing import Process

import numpy as np

from client import Client
from server_nis import TrainServer, ExperimentConfig, Mode

np.random.seed(42)
TEST_DATA = np.random.rand(1000, 1000).astype(np.float32)


def test_nis_client_server():
    client_process = Process(target=client_process_func)
    server_process = Process(target=server_process_func)
    server_process.start()
    client_process.start()
    client_process.join()
    server_process.join()


def server_process_func():
    server = TrainServer(ExperimentConfig(num_context_features=8, save_plots=False))
    server.nis.initialize(mode='server')
    server.connect()

    # inference hybrid_sampling
    server.hybrid_sampling = True
    server.mode = Mode.INFERENCE
    server.connection.send(server.put_infer_ok.name)
    server.receive_length()
    server.receive_raw()
    server.process()

    # inference nis
    server.nis.num_frame = 2
    server.nis.train_sampling_call_difference = 0
    server.hybrid_sampling = False
    server.mode = Mode.INFERENCE
    server.connection.send(server.put_infer_ok.name)
    server.receive_length()
    server.receive_raw()
    server.process()

    # train nis
    server.hybrid_sampling = False
    server.mode = Mode.TRAIN
    server.connection.send(server.put_train_ok.name)
    server.receive_length()
    server.receive_raw()
    server.process()


def client_process_func():
    client = Client()
    client.connect()

    # inference hybrid_sampling
    answer = client.client_socket.recv(client.put_infer_ok.length)
    if answer == client.put_infer_ok.name:
        client.get_samples()

    # inference nis
    answer = client.client_socket.recv(client.put_infer_ok.length)
    if answer == client.put_infer_ok.name:
        client.get_samples()

    # train nis
    answer = client.client_socket.recv(client.put_train_ok.length)
    if answer == client.put_train_ok.name:
        client.send_radiance()
