import signal
import socket
import json
import sys
import time
from offline_client import OfflineClient
from logger import logger


class SocketServer:
    def __init__(self, *args, **kwargs):
        self.keep_alive = kwargs.get("keep_alive", False)

        self.start_server()

    def start_server(self):
        try:
            self.start()
            self.run()
        except TimeoutError:
            self.stop()
            if self.keep_alive:
                self.start_server()

    def start(self):
        logger.info("Initializing SocketServer")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # set SO_REUSEADDR option
        self.server_socket.bind(('127.0.0.1', 5000))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1)
        logger.info("Server listening on port 5000")
        signal.signal(signal.SIGINT, self.close_server)

    def run(self):
        self.running = True
        self.client = OfflineClient(socket_server=self)
        # set a timeout on client_socket.recv so that we can check if the server is still running
        (self.client_socket, self.client_address) = self.server_socket.accept()
        self.client_socket.settimeout(1)
        logger.info(f"Client connected from {self.client_address}")
        packets = []
        while self.running:
            try:
                data = self.client_socket.recv(1024)
                if data == b'':
                    break
                # get data in 1024 byte packets until we get a 1024 byte zero chunk
                if data == b'\x00' * 1024:
                    # we have received the end of the message
                    logger.info("Received end message")
                    data = b''.join(packets)
                    # strip x00 padding
                    data = data.rstrip(b'\x00')
                    packets = []
                    try:
                        self.client.message = json.loads(data.decode("utf-8"))
                    except json.decoder.JSONDecodeError:
                        logger.error("Invalid json in request")
                    continue
                else:
                    packets.append(data)
            except socket.timeout:
                continue
            time.sleep(0.01)
        self.start_server()
    byte_packet_size = 1024

    def process_response(self, response: dict):
        # convert response to json string
        response = json.dumps(response)
        # convert response to bytes
        response = response.encode()
        # pad the response if it is less than byte_packet_size
        response = self.pad_packet(response)
        # create the packets
        packets = [response[i:i + self.byte_packet_size] for i in range(0, len(response), self.byte_packet_size)]
        return packets

    def pad_packet(self, packet):
        if len(packet) < self.byte_packet_size:
            packet += b' ' * (self.byte_packet_size - len(packet))
        return packet

    def send_response(self, response):
        packets = self.process_response(response)
        for packet in packets:
            packet = self.pad_packet(packet)
            self.client_socket.sendall(packet)
        self.send_end_message()

    def send_end_message(self):
        # send a 1024 byte zero chunk to indicate the end of the message
        self.client_socket.sendall(b'\x00' * 1024)

    def stop(self):
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()
        self.running = False
        time.sleep(2)

    client_socket = None
    def close_server(self, signal, frame):
        logger.info("Closing server")
        self.stop()
        sys.exit(0)


if __name__ == "__main__":
    SocketServer()
