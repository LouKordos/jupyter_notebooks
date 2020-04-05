import socket
from casadi import *
import numpy as np

opts = {}
opts["print_time"] = 0
#opts["expand"] = False
opts['ipopt'] = {"max_iter":50, "print_level":0, "acceptable_tol":1e-7, "acceptable_obj_change_tol":1e-5}

solver = nlpsol("solver", "ipopt", "nlp.so", opts)

lbg = []
ubg = []
lbx = []
ubx = []

n = 13
m = 6
N = 30

f_min = -500
f_max = 500

constraints_length = n + N * n + int((N / m)) * m + int((N / m)) * 8
bounds_length = n * (N+1) + m * N
decision_variables_length = n * (N+1) + m * N

print("Constraints length:", constraints_length)
print("Bounds length:", bounds_length)
print("Decision variables length:", decision_variables_length)

print(int(N / m))
print(constraints_length - int(N / m) * 8)

for i in range(constraints_length - int((N / m)) * 8):
	lbg += [0]

for i in range(constraints_length - int((N / m)) * 8, constraints_length):
	lbg += [-inf]

print(len(lbg))

for i in range(constraints_length):
	ubg += [0]

for i in range(n * (N+1)):
	lbx += [-inf]
	ubx += [inf]


for i in range(n * (N+1), bounds_length):
	lbx += [f_min]
	ubx += [f_max]

x_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

U_t = np.zeros((m, N))
X_t = np.tile(np.array(x_t).reshape(n, 1), N+1).reshape(n, N+1)

X_t = X_t.reshape((n * (N+1), 1))
U_t = U_t.reshape((m * N, 1))

x_ref = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -9.81]

UDP_IP = "127.0.0.1"
UDP_PORT = 6969

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

sock.bind((UDP_IP, UDP_PORT))

import time

while True:
	start_time = time.time()
	state_str, addr = sock.recvfrom(4096)
	states_split = state_str.decode().split('|')
	x_t = [float(states_split[0]), float(states_split[1]), float(states_split[2]), float(states_split[3]), float(states_split[4]), float(states_split[5]), float(states_split[6]), float(states_split[7]), float(states_split[8]), float(states_split[9]), float(states_split[10]), float(states_split[11]), float(states_split[12])]
	#print("x_t:", x_t)
	x_t = DM(x_t)
	x0_solver = vertcat(*[X_t, U_t])
	sol = solver(x0=x0_solver, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=vertcat(*[x_t, x_ref]))
	u_t = sol['x'][n * (N+1):n * (N+1) + m]
	sock.sendto(bytes("{0}|{1}|{2}|{3}|{4}|{5}".format(u_t[0], u_t[1], u_t[2], u_t[3], u_t[4], u_t[5]), "utf-8"), addr)
	#print("U_t:", u_t)
	u_t = u_t = sol['x'][n * (N+1):n * (N+1) + m]

	X_t[:-n] = sol['x'][n:n * (N+1)]
	X_t[-n:] = sol['x'][-n - m * N: n * (N+1)]

	U_t[:-m] = sol['x'][m + n * (N+1):]
	U_t[-m:] = sol['x'][-m:]

	end_time = time.time()
	print(x_t)
	print("Iteration time:", (end_time - start_time))

sock.sendto(bytes(message, "utf-8"), (UDP_IP, UDP_PORT))