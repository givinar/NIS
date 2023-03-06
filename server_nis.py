import argparse
import sys
import numpy as np
import torch

from experiment_config import ExperimentConfig
from nis import NeuralImportanceSampling
from utils import pyhocon_wrapper, utils

# Using by server
import socket
from enum import Enum
from collections import namedtuple
import logging

Request = namedtuple("Request", "name length")


class Mode(Enum):
    UNKNOWN = -1
    TRAIN = 0
    INFERENCE = 1


class TrainServer:
    def __init__(self, _config: ExperimentConfig):
        self.config = _config
        self.host = self.config.host
        self.port = self.config.port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.NUM_CONNECTIONS = 5
        self.BUFSIZE = 8192
        self.connection = None
        self.put_infer = Request(b"PUT INFER", 9)
        self.put_train = Request(b"PUT TRAIN", 9)
        self.put_infer_ok = Request(b"PUT INFER OK", 12)
        self.put_train_ok = Request(b"PUT TRAIN OK", 12)
        self.data_ok = Request(b"DATA OK", 7)

        self.length = 0
        self.raw_data = bytearray()
        self.data = bytearray()
        self.mode = Mode

        self.sock.bind((self.host, self.port))
        self.sock.listen(self.NUM_CONNECTIONS)
        self.nis = NeuralImportanceSampling(_config)
        self.hybrid_sampling = self.config.hybrid_sampling

        self.samples_tensor = None

    def connect(self):
        print(f"Waiting for connection by {self.host}")
        self.connection, address = self.sock.accept()
        print(f"Connected by {self.host} successfully")

    def close(self):
        self.connection.close()

    def receive_length(self):
        self.length = int.from_bytes(self.connection.recv(4), "little")
        # print(str(self.length))

    def receive_raw(self):
        self.raw_data = bytearray()
        bytes_recd = 0
        while bytes_recd < self.length:
            chunk = self.connection.recv(min(self.length - bytes_recd, self.BUFSIZE))
            if chunk == b"":
                raise RuntimeError("socket connection broken")
            self.raw_data.extend(chunk)
            bytes_recd = bytes_recd + len(chunk)

    def send(self, data):
        try:
            self.connection.sendall(data)
        except ConnectionError:
            logging.error("Client was disconnected suddenly while sending\n")

    def make_infer(self):
        self.nis.train_sampling_call_difference += 1
        if self.nis.train_sampling_call_difference == 1:
            self.nis.num_frame += 1
            print("Frame num: " + str(self.nis.num_frame))
        # points = np.frombuffer(self.raw_data, dtype=np.float32).reshape((-1, self.config.num_context_features + 2)) #add vec2 light_sample_dir
        points = np.frombuffer(self.raw_data, dtype=np.float32).reshape(
            (-1, 8 + 2)
        )  # Temporal solution for ignoring processing additional inputs in NN
        if self.hybrid_sampling:
            samples, pdfs, pdf_light_samples, coef = self._make_infer_hybrid(points)
        else:
            if (self.nis.num_frame != 1) and (
                not self.config.one_bounce_mode
                or (self.nis.train_sampling_call_difference == 1)
            ):
                samples, pdfs, pdf_light_samples, coef = self._make_infer_nis(points)
            else:
                samples, pdfs, pdf_light_samples, coef = self._make_infer_hybrid(points)
        return [samples, pdf_light_samples, pdfs, coef]
        # return [samples, pdf_light_samples, torch.nn.Softmax()(pdfs), coef]

    def _make_infer_hybrid(self, points):
        pdf_light_samples = utils.get_pdf_by_samples_cosine(points[:, 8:])
        # [samples, pdfs] = utils.get_test_samples_cosine(
        #     points
        # )
        [samples, pdfs] = utils.get_test_samples_vectorized(points)# lights(vec3), pdfs
        # pdf_light_samples = utils.get_pdf_by_samples_uniform(points[:, 8:])
        # [samples, pdfs] = utils.get_test_samples_uniform(points)  # lights(vec3), pdfs
        coef = torch.ones(pdf_light_samples.size())
        return samples, pdfs, pdf_light_samples, coef

    def _make_infer_nis(self, points):
        [samples, pdf_light_samples, pdfs, coef] = self.nis.get_samples(points)
        self.samples_tensor = (
            samples.clone().numpy()
        )  # This is only needed for the make_train step and pass it to the Gaussian function.
        samples[:, 0] = samples[:, 0] * 2 * np.pi
        samples[:, 1] = torch.acos(samples[:, 1])

        # MIS should be implemented here
        # pdfs = (1 / (2 * np.pi)) / pdfs
        # pdf_light_samples = pdf_light_samples / (2 * np.pi)
        # if self.nis.num_frame < 100:
        #    pdfs = torch.ones(pdfs.size())
        #    pdfs /= (2 * np.pi)
        #    pdf_light_samples = torch.ones(pdfs.size())
        #    pdf_light_samples /= (2 * np.pi)
        return samples, pdfs, pdf_light_samples, coef

    def make_train(self):
        # context = np.frombuffer(self.raw_data, dtype=np.float32).reshape((-1, self.config.num_context_features + 3))
        context = np.frombuffer(self.raw_data, dtype=np.float32).reshape((-1, 8 + 3 + 1))
        context = context[~np.isnan(context).any(axis=1), :]
        if self.hybrid_sampling:
            pass
        else:
            if (self.nis.num_frame != 1) and (
                not self.config.one_bounce_mode
                or (self.nis.train_sampling_call_difference == 1)
            ):
                lum = 0.3 * context[:, 0] + 0.3 * context[:, 1] + 0.3 * context[:, 2]
                # Checking the Gaussian distribution
                # y = self.nis.function(torch.from_numpy(self.samples_tensor))
                # lum[0] = y[0].item()
                # lum[1] = y[1].item()
                # lum[2] = y[2].item()
                tdata = context[:, [3, 4, 5, 6, 7, 8, 9, 10, 11]]
                tdata = np.concatenate(
                    (tdata, lum.reshape([len(lum), 1])), axis=1, dtype=np.float32
                )
                self.nis.train(context=tdata)
            else:
                pass
        self.nis.train_sampling_call_difference -= 1

    def process(self):
        try:
            logging.debug("Mode = %s", self.mode.name)
            logging.debug(
                "Len = %s, Data = %s",
                self.length,
                np.frombuffer(self.raw_data, dtype=np.float32),
            )
            if self.mode == Mode.TRAIN:
                self.make_train()
                self.connection.send(self.data_ok.name)
            elif self.mode == Mode.INFERENCE:
                [samples, pdf_light_samples, pdfs, coef] = self.make_infer()
                self.connection.send(self.put_infer.name)
                answer = self.connection.recv(self.put_infer_ok.length)
                if answer == self.put_infer_ok.name:
                    raw_data = bytearray()
                    s = samples.cpu().detach().numpy()
                    pls = pdf_light_samples.cpu().detach().numpy()
                    pls[pls < 0] = 0
                    s = np.concatenate(
                        (s, pls.reshape([len(pls), 1])), axis=1, dtype=np.float32
                    )
                    p = pdfs.cpu().detach().numpy().reshape([-1, 1])
                    c = coef.cpu().detach().numpy().reshape([-1, 1])
                    raw_data.extend(np.concatenate((s, p, c), axis=1).tobytes())

                    self.connection.send(len(raw_data).to_bytes(4, "little"))
                    self.connection.sendall(raw_data)
                    answer = self.connection.recv(self.data_ok.length)
                    if answer == self.data_ok.name:
                        logging.info("Inference data was sent successfully ...\n")
                    else:
                        logging.error("Inference data wasn't sent ...\n")
                else:
                    logging.error("Answer is not equal " + self.put_ok.name)
            else:
                logging.error("Unknown packet type ...")

        except ConnectionError:
            logging.error("Connection failed ...")

    def run(self):
        self.nis.initialize(mode="server")
        self.connect()
        try:
            logging.debug("Server started ...")
            while True:
                cmd = self.connection.recv(self.put_infer.length)
                if cmd == self.put_infer.name:
                    self.mode = Mode.INFERENCE
                    self.connection.send(self.put_infer_ok.name)
                    self.receive_length()
                    self.receive_raw()
                    self.process()
                elif cmd == self.put_train.name:
                    self.mode = Mode.TRAIN
                    self.connection.send(self.put_train_ok.name)
                    self.receive_length()
                    self.receive_raw()
                    self.process()

        except ConnectionError:
            logging.error("Connection failed ...")
        finally:
            self.close()


def parse_args(arg=sys.argv[1:]):
    train_parser = argparse.ArgumentParser(description="Application for model training")

    train_parser.add_argument(
        "-c", "--config", required=True, help="Configuration file path"
    )

    return train_parser.parse_args(arg)


def server_processing(experiment_config):
    server = TrainServer(experiment_config)
    server.run()


if __name__ == "__main__":
    options = parse_args()
    config = pyhocon_wrapper.parse_file(options.config)
    experiment_config = ExperimentConfig.init_from_pyhocon(config)

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)
    server_processing(experiment_config)
