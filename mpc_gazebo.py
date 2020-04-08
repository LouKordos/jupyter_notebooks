import socket
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from sympy import *
from sympy.physics.mechanics import dynamicsymbols, init_vprinting, msubs
from IPython.display import Image
from IPython.core.display import HTML
import scipy.integrate
import math
from numpy.linalg import matrix_power
from scipy.linalg import expm
from casadi import *

UDP_IP = "127.0.0.1"
UDP_PORT = 6969

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

sock.bind((UDP_IP, UDP_PORT))

def discretize_ss(A, B, dt):
    A_B = np.block([[A, B],
                [np.zeros((B.shape[1], A.shape[0])), np.zeros((B.shape[1], B.shape[1]))]])
    #print("A shape:", A.shape)
    #print("B shape:", B.shape)
    #print("A_B:", A_B)

    eAt_d = scipy.linalg.expm(A_B * dt)

    A_d_temp = eAt_d[:A.shape[0], :A.shape[0]]

    B_d_temp = eAt_d[:B.shape[0], A.shape[0]:]

    return (A_d_temp, B_d_temp)

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

dt = 1/30

f_min = -1000
f_max = 1000

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

x_t = [0, 0, 0, 0, 0, 1.48, 0, 0, 0, 0, 0, 0, 0]

U_t = np.zeros((m, N))
X_t = np.tile(np.array(x_t).reshape(n, 1), N+1).reshape(n, N+1)

X_t = X_t.reshape((n * (N+1), 1))
U_t = U_t.reshape((m * N, 1))

pos_desired = 1
vel_desired = 0.01

x_ref = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -9.81]

x_ref = np.tile(np.array(x_ref).reshape(n, 1), N)

P_rows = n
P_cols = 1 + N + n * N + m * N + m # last m for D matrix (used for contact constraint)

P_param = np.zeros((P_rows, P_cols))
P_param[:, 0] = np.array(x_t)
P_param[:, 1:N+1] = x_ref

swing_left = True # Swing means in air, thus no contact
swing_right = False

D = np.array([[int(swing_left == True), 0, 0, 0, 0, 0],
		 			  [0, int(swing_left == True), 0, 0, 0, 0],
					  [0, 0, int(swing_left == True), 0, 0, 0],
					  [0, 0, 0, int(swing_right == True), 0, 0],
					  [0, 0, 0, 0, int(swing_right == True), 0],
					  [0, 0, 0, 0, 0, int(swing_right == True)]]) #Swing = no contact (and thus 1 any force == 0 must mean force is 0)

P_param[:m, 1+N+n*N + m*N:1+N+n*N + m*N + m] = D

P_param = DM(P_param)

m_value = 30 # [kg]
    
I_body = np.array([[0.1536, 0., 0.],
					[0., 0.6288, 0.],
					[0., 0., 0.6843]]) # Inertia in the body frame (meaning around CoM of torso).

Ixx = I_body[0, 0]
Ixy = I_body[0, 1]
Ixz = I_body[0, 2]

Iyx = I_body[1, 0]
Iyy = I_body[1, 1]
Iyz = I_body[1, 2]

Izx = I_body[2, 0]
Izy = I_body[2, 1]
Izz = I_body[2, 2]

t = 0
iterations = 0

dt_step = .5 # seconds between gait phase change

foot_behind = True

pos_desired = 0
vel_desired = 0.1

while True:
	start_time = time.time()

	state_str, addr = sock.recvfrom(4096)
	states_split = state_str.decode().split('|')
	x_t = [float(states_split[0]), float(states_split[1]), float(states_split[2]), float(states_split[3]), float(states_split[4]), float(states_split[5]), float(states_split[6]), float(states_split[7]), float(states_split[8]), float(states_split[9]), float(states_split[10]), float(states_split[11]), float(states_split[12])]
	#print("x_t:", x_t)
	x_t = DM(x_t)

	setup_start_time = time.time()

	if iterations % 8 == 0 and True: # Assuming it runs at 20Hz for now
		swing_left = not swing_left
		swing_right = not swing_right
		D = np.array([[int(swing_left == True), 0, 0, 0, 0, 0],
		 			  [0, int(swing_left == True), 0, 0, 0, 0],
					  [0, 0, int(swing_left == True), 0, 0, 0],
					  [0, 0, 0, int(swing_right == True), 0, 0],
					  [0, 0, 0, 0, int(swing_right == True), 0],
					  [0, 0, 0, 0, 0, int(swing_right == True)]]) #Swing = no contact (and thus 1 any force == 0 must mean force is 0)

		print(D)
		P_param[:m, 1+N+n*N + m*N:1+N+n*N + m*N + m] = D

		if not swing_left: # Contact on left foot
			if foot_behind:
				r_y_left = x_t[4] - 0.1
			else:
				r_y_left = x_t[4] + 0.1
        
		if not swing_right:
			if foot_behind:
				r_y_right = x_t[4] - 0.1
			else:
				r_y_right = x_t[4] + 0.1
                
		foot_behind = not foot_behind
	
	P_param[:, 0] = x_t

	pos_desired += vel_desired * dt
	for i in range(N):
		x_ref[:, i] = np.array([0, 0, 0, 0, pos_desired, 1.48, 0, 0, 0, 0, vel_desired, 0, -9.81])

	P_param[:, 1:N+1] = x_ref


	for i in range(N):

		phi_t = x_ref[0, i]
		theta_t = x_ref[1, i]
		psi_t = x_ref[2, i]

		I_world = np.array([[(Ixx*cos(psi_t) + Iyx*sin(psi_t))*cos(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*sin(psi_t), -(Ixx*cos(psi_t) + Iyx*sin(psi_t))*sin(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*cos(psi_t), Ixz*cos(psi_t) + Iyz*sin(psi_t)], [(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*cos(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*sin(psi_t), -(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*sin(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*cos(psi_t), -Ixz*sin(psi_t) + Iyz*cos(psi_t)], [Ixy*sin(psi_t) + Izx*cos(psi_t), Ixy*cos(psi_t) - Izx*sin(psi_t), Izz]])
		# Location of the force vector being applied by the left foot.
		r_x_left = 0.15
		r_y_left = 0
		r_z_left = 0

		# Location of the force vector being applied by the right foot.
		r_x_right = -0.15
		r_y_right = 0
		r_z_right = 0

		# Skew symmetric versions for the 3x1 foot position vector resembling the matrix version of the cross product of two vectors. This is needed for the matrix form.
		r_left_skew_symmetric = np.array([[0, -r_z_left, r_y_left],
										[r_z_left, 0, -r_x_left],
										[-r_y_left, r_x_left, 0]]) 

		r_right_skew_symmetric = np.array([[0, -r_z_right, r_y_right],
										[r_z_right, 0, -r_x_right],
										[-r_y_right, r_x_right, 0]])

		A_c = np.array([[0, 0, 0, 0, 0, 0, math.cos(psi_t), math.sin(psi_t), 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, -math.sin(psi_t), math.cos(psi_t), 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],

							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],

							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

		B_c = np.block([[0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0],
						[I_world @ r_left_skew_symmetric, I_world @ r_right_skew_symmetric],
						[1/m_value, 0, 0, 1/m_value, 0, 0],
						[0, 1/m_value, 0, 0, 1/m_value, 0],
						[0, 0, 1/m_value, 0, 0, 1/m_value],
						[0, 0, 0, 0, 0, 0]])

		A_d, B_d = discretize_ss(A_c, B_c, dt)
		
		P_param[:, 1 + N + (i*n):1 + N + (i*n)+n] = A_d
		P_param[:, 1 + N + n * N + (i*m):1 + N + n * N + (i*m)+m] = B_d

	setup_end_time = time.time()
	#print("Setup time:", (setup_end_time - setup_start_time))
	x0_solver = vertcat(*[X_t, U_t])
	sol = solver(x0=x0_solver, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=P_param)
	u_t = sol['x'][n * (N+1):n * (N+1) + m]
	#print(r_z_left)
	sock.sendto(bytes("{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}".format(u_t[0], u_t[1], u_t[2], u_t[3], u_t[4], u_t[5], r_x_left, r_y_left, r_z_left, r_x_right, r_y_right, r_z_right), "utf-8"), addr)

	X_t[:-n] = sol['x'][n:n * (N+1)]
	X_t[-n:] = sol['x'][-n - m * N: n * (N+1)]

	U_t[:-m] = sol['x'][m + n * (N+1):]
	U_t[-m:] = sol['x'][-m:]

	t += dt
	iterations += 1

	end_time = time.time()

	print("Loop frequency:", 1/(end_time - start_time), "Hz")
	#print("Iteration time:", (end_time - start_time))
	#print("Rest of the iteration (without setup):", (end_time - start_time) - (setup_end_time - setup_start_time))