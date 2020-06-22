import os
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 6969

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

#sock.bind((UDP_IP, UDP_PORT))
i = 0
while True:
    sock.sendto(bytes("{0}|0|0|0|0|0".format(i), "utf-8"), (UDP_IP, UDP_PORT))
    state_str, addr = sock.recvfrom(4096)
    print(state_str)
    print(addr)
    i += 1