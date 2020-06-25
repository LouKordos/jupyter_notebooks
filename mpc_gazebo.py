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

udp_ip = "127.0.0.1"
mpc_port = 4801

left_leg_port = 4200
right_leg_port = 4201

mpc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mpc_socket.bind((udp_ip, mpc_port))

left_leg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
left_leg_socket.bind((udp_ip, left_leg_port))

right_leg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_leg_socket.bind((udp_ip, right_leg_port))

def discretize_ss(A, B, dt):
    A_B = np.block([[A, B],
                [np.zeros((B.shape[1], A.shape[0])), np.zeros((B.shape[1], B.shape[1]))]])

    eAt_d = scipy.linalg.expm(A_B * dt)

    A_d_temp = eAt_d[:A.shape[0], :A.shape[0]]

    B_d_temp = eAt_d[:B.shape[0], A.shape[0]:]

    return (A_d_temp, B_d_temp)

opts = {}
opts["print_time"] = 0
#opts["expand"] = False
opts['ipopt'] = {"max_iter":40, "print_level":0, "acceptable_tol":1e-7, "acceptable_obj_change_tol":1e-5}

solver = nlpsol("solver", "ipopt", "nlp.so", opts)

lbg = []
ubg = []
lbx = []
ubx = []

n = 13
m = 6
N = 20

dt = 1/30.0

f_min_z = 0
f_max_z = 700

lbg += [0] * n
ubg += [0] * n

# Dynamics constraints
for i in range(N):
    lbg += [0] * n
    ubg += [0] * n

# Contact constraints
for i in range(N):
    lbg += [0] * m
    ubg += [0] * m

for i in range(N):
	lbg += [-inf] * 8
	ubg += [inf] * 8

for i in range(n*(N+1)):
    lbx += [-inf]
    ubx += [inf]

for i in range(N):
	lbx += [-inf, -inf, f_min_z, -inf, -inf, f_min_z]
	ubx += [inf, inf, f_max_z, inf, inf, f_max_z]

x_t = [0, 0, 0, 0, 0, 1.48, 0, 0, 0, 0, 0, 0, -9.81]

U_t = np.zeros((m, N))
U_t = U_t.reshape((m * N, 1))
X_t = np.matlib.repmat(np.array(x_t).reshape(n,1), N+1, 1)#.reshape(n*(N+1), 1) # np.tile(np.array(x_t).reshape(n, 1), N+1)

x_ref = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -9.81]

x_ref = np.tile(np.array(x_ref).reshape(n, 1), N)

P_rows = n
P_cols = 1 + N + n * N + m * N + N * m 

P_param = np.zeros((P_rows, P_cols))
P_param[:, 1:N+1] = x_ref

swing_left = False
swing_right = False

m_value = 30 # [kg]

I_body = np.array([[0.1536, 0., 0.],
                    [0., 0.62288, 0.],
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

contact_swap_interval = 15

foot_behind_left = swing_left # this is because it will be inverted on the first contact switch
foot_behind_right = swing_right

pos_y_desired = 0
vel_y_desired = 0

pos_z_desired = 0.8
vel_z_desired = 0

psi_desired = 0
omega_z_desired = 0

step_length = 0

r_y_left = r_y_right = 0
r_x_left = -0.15
r_x_right = 0.15

legs_attached = False

for i in range(0, 10):
	sol = solver(x0=np.zeros((n*(N+1)+m*N, 1)), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=P_param)

print("Entering main MPC loop...")

log_file = open("../cpp/walking_controller/plot_data/mpc_log.csv", 'w')
log_file.write("t,phi,theta,psi,pos_x,pos_y,pos_z,omega_x,omega_y,omega_z,vel_x,vel_y,vel_z,g,f_x_left,f_y_left,f_z_left,f_x_right,f_y_right,f_z_right,r_x_left,r_y_left,r_z_left,r_x_right,r_y_right,r_z_right,x_t_temp_theta\n")
log_file.close()
log_file = open("../cpp/walking_controller/plot_data/mpc_log.csv", "a")

control_history = [np.zeros((m,1)), np.zeros((m,1))]

def get_joint_torques(f_x, f_y, f_z, theta1, theta2, theta3, theta4, theta5, phi, theta, psi):
    return np.array([[f_x*(0.41*sin(theta2)*sin(psi + theta1)*cos(theta3) + 0.4*sin(theta2)*sin(psi + theta1)*cos(theta3 + theta4) + 0.04*sin(theta2)*sin(psi + theta1)*cos(theta3 + theta4 + theta5) - 0.41*sin(theta3)*cos(psi + theta1) - 0.4*sin(theta3 + theta4)*cos(psi + theta1) - 0.04*sin(theta3 + theta4 + theta5)*cos(psi + theta1))*cos(theta) + f_y*((sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))*(0.41*sin(theta1)*sin(theta3) + 0.4*sin(theta1)*sin(theta3 + theta4) + 0.04*sin(theta1)*sin(theta3 + theta4 + theta5) + 0.41*sin(theta2)*cos(theta1)*cos(theta3) + 0.4*sin(theta2)*cos(theta1)*cos(theta3 + theta4) + 0.04*sin(theta2)*cos(theta1)*cos(theta3 + theta4 + theta5)) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(0.41*sin(theta1)*sin(theta2)*cos(theta3) + 0.4*sin(theta1)*sin(theta2)*cos(theta3 + theta4) + 0.04*sin(theta1)*sin(theta2)*cos(theta3 + theta4 + theta5) - 0.41*sin(theta3)*cos(theta1) - 0.4*sin(theta3 + theta4)*cos(theta1) - 0.04*sin(theta3 + theta4 + theta5)*cos(theta1))) + f_z*((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(0.41*sin(theta1)*sin(theta2)*cos(theta3) + 0.4*sin(theta1)*sin(theta2)*cos(theta3 + theta4) + 0.04*sin(theta1)*sin(theta2)*cos(theta3 + theta4 + theta5) - 0.41*sin(theta3)*cos(theta1) - 0.4*sin(theta3 + theta4)*cos(theta1) - 0.04*sin(theta3 + theta4 + theta5)*cos(theta1)) - (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(0.41*sin(theta1)*sin(theta3) + 0.4*sin(theta1)*sin(theta3 + theta4) + 0.04*sin(theta1)*sin(theta3 + theta4 + theta5) + 0.41*sin(theta2)*cos(theta1)*cos(theta3) + 0.4*sin(theta2)*cos(theta1)*cos(theta3 + theta4) + 0.04*sin(theta2)*cos(theta1)*cos(theta3 + theta4 + theta5)))], [f_x*((0.41*sin(theta2)*cos(theta3) + 0.4*sin(theta2)*cos(theta3 + theta4) + 0.04*sin(theta2)*cos(theta3 + theta4 + theta5))*sin(theta) + (0.41*cos(theta2)*cos(theta3) + 0.4*cos(theta2)*cos(theta3 + theta4) + 0.04*cos(theta2)*cos(theta3 + theta4 + theta5))*sin(psi)*sin(theta1)*cos(theta) - (0.41*cos(theta2)*cos(theta3) + 0.4*cos(theta2)*cos(theta3 + theta4) + 0.04*cos(theta2)*cos(theta3 + theta4 + theta5))*cos(psi)*cos(theta)*cos(theta1)) - f_y*(-(sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))*(0.41*cos(theta2)*cos(theta3) + 0.4*cos(theta2)*cos(theta3 + theta4) + 0.04*cos(theta2)*cos(theta3 + theta4 + theta5))*sin(theta1) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(0.41*cos(theta2)*cos(theta3) + 0.4*cos(theta2)*cos(theta3 + theta4) + 0.04*cos(theta2)*cos(theta3 + theta4 + theta5))*cos(theta1) + (0.41*sin(theta2)*cos(theta3) + 0.4*sin(theta2)*cos(theta3 + theta4) + 0.04*sin(theta2)*cos(theta3 + theta4 + theta5))*sin(phi)*cos(theta)) - f_z*((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(0.41*cos(theta2)*cos(theta3) + 0.4*cos(theta2)*cos(theta3 + theta4) + 0.04*cos(theta2)*cos(theta3 + theta4 + theta5))*cos(theta1) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(0.41*cos(theta2)*cos(theta3) + 0.4*cos(theta2)*cos(theta3 + theta4) + 0.04*cos(theta2)*cos(theta3 + theta4 + theta5))*sin(theta1) - (0.41*sin(theta2)*cos(theta3) + 0.4*sin(theta2)*cos(theta3 + theta4) + 0.04*sin(theta2)*cos(theta3 + theta4 + theta5))*cos(phi)*cos(theta))], [-f_x*(-0.41*sin(theta)*sin(theta3)*cos(theta2) - 0.4*sin(theta)*sin(theta3 + theta4)*cos(theta2) - 0.04*sin(theta)*sin(theta3 + theta4 + theta5)*cos(theta2) - 0.41*sin(theta2)*sin(theta3)*cos(theta)*cos(psi + theta1) - 0.4*sin(theta2)*sin(theta3 + theta4)*cos(theta)*cos(psi + theta1) - 0.04*sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta)*cos(psi + theta1) + 0.41*sin(psi + theta1)*cos(theta)*cos(theta3) + 0.4*sin(psi + theta1)*cos(theta)*cos(theta3 + theta4) + 0.04*sin(psi + theta1)*cos(theta)*cos(theta3 + theta4 + theta5)) - f_y*((sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))*(0.41*sin(theta1)*sin(theta2)*sin(theta3) + 0.4*sin(theta1)*sin(theta2)*sin(theta3 + theta4) + 0.04*sin(theta1)*sin(theta2)*sin(theta3 + theta4 + theta5) + 0.41*cos(theta1)*cos(theta3) + 0.4*cos(theta1)*cos(theta3 + theta4) + 0.04*cos(theta1)*cos(theta3 + theta4 + theta5)) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(0.41*sin(theta1)*cos(theta3) + 0.4*sin(theta1)*cos(theta3 + theta4) + 0.04*sin(theta1)*cos(theta3 + theta4 + theta5) - 0.41*sin(theta2)*sin(theta3)*cos(theta1) - 0.4*sin(theta2)*sin(theta3 + theta4)*cos(theta1) - 0.04*sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta1)) - (-0.41*sin(theta3) - 0.4*sin(theta3 + theta4) - 0.04*sin(theta3 + theta4 + theta5))*sin(phi)*cos(theta)*cos(theta2)) - f_z*((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(0.41*sin(theta1)*cos(theta3) + 0.4*sin(theta1)*cos(theta3 + theta4) + 0.04*sin(theta1)*cos(theta3 + theta4 + theta5) - 0.41*sin(theta2)*sin(theta3)*cos(theta1) - 0.4*sin(theta2)*sin(theta3 + theta4)*cos(theta1) - 0.04*sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta1)) - (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(0.41*sin(theta1)*sin(theta2)*sin(theta3) + 0.4*sin(theta1)*sin(theta2)*sin(theta3 + theta4) + 0.04*sin(theta1)*sin(theta2)*sin(theta3 + theta4 + theta5) + 0.41*cos(theta1)*cos(theta3) + 0.4*cos(theta1)*cos(theta3 + theta4) + 0.04*cos(theta1)*cos(theta3 + theta4 + theta5)) + (-0.41*sin(theta3) - 0.4*sin(theta3 + theta4) - 0.04*sin(theta3 + theta4 + theta5))*cos(phi)*cos(theta)*cos(theta2))], [-f_x*(-0.4*sin(theta)*sin(theta3 + theta4)*cos(theta2) - 0.04*sin(theta)*sin(theta3 + theta4 + theta5)*cos(theta2) - 0.4*sin(theta2)*sin(theta3 + theta4)*cos(theta)*cos(psi + theta1) - 0.04*sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta)*cos(psi + theta1) + 0.4*sin(psi + theta1)*cos(theta)*cos(theta3 + theta4) + 0.04*sin(psi + theta1)*cos(theta)*cos(theta3 + theta4 + theta5)) - f_y*((sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))*(0.4*sin(theta1)*sin(theta2)*sin(theta3 + theta4) + 0.04*sin(theta1)*sin(theta2)*sin(theta3 + theta4 + theta5) + 0.4*cos(theta1)*cos(theta3 + theta4) + 0.04*cos(theta1)*cos(theta3 + theta4 + theta5)) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(0.4*sin(theta1)*cos(theta3 + theta4) + 0.04*sin(theta1)*cos(theta3 + theta4 + theta5) - 0.4*sin(theta2)*sin(theta3 + theta4)*cos(theta1) - 0.04*sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta1)) - (-0.4*sin(theta3 + theta4) - 0.04*sin(theta3 + theta4 + theta5))*sin(phi)*cos(theta)*cos(theta2)) - f_z*((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(0.4*sin(theta1)*cos(theta3 + theta4) + 0.04*sin(theta1)*cos(theta3 + theta4 + theta5) - 0.4*sin(theta2)*sin(theta3 + theta4)*cos(theta1) - 0.04*sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta1)) - (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(0.4*sin(theta1)*sin(theta2)*sin(theta3 + theta4) + 0.04*sin(theta1)*sin(theta2)*sin(theta3 + theta4 + theta5) + 0.4*cos(theta1)*cos(theta3 + theta4) + 0.04*cos(theta1)*cos(theta3 + theta4 + theta5)) + (-0.4*sin(theta3 + theta4) - 0.04*sin(theta3 + theta4 + theta5))*cos(phi)*cos(theta)*cos(theta2))], [-f_x*(-0.04*sin(theta)*sin(theta3 + theta4 + theta5)*cos(theta2) - 0.04*sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta)*cos(psi + theta1) + 0.04*sin(psi + theta1)*cos(theta)*cos(theta3 + theta4 + theta5)) - f_y*(-0.04*(-sin(theta1)*cos(theta3 + theta4 + theta5) + sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta1))*(sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi)) + 0.04*(sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))*(sin(theta1)*sin(theta2)*sin(theta3 + theta4 + theta5) + cos(theta1)*cos(theta3 + theta4 + theta5)) + 0.04*sin(phi)*sin(theta3 + theta4 + theta5)*cos(theta)*cos(theta2)) - f_z*(-0.04*(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(-sin(theta1)*cos(theta3 + theta4 + theta5) + sin(theta2)*sin(theta3 + theta4 + theta5)*cos(theta1)) - 0.04*(sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(sin(theta1)*sin(theta2)*sin(theta3 + theta4 + theta5) + cos(theta1)*cos(theta3 + theta4 + theta5)) - 0.04*sin(theta3 + theta4 + theta5)*cos(phi)*cos(theta)*cos(theta2))]])

while True:
	start_time = time.time()

	state_str, mpc_addr = mpc_socket.recvfrom(4096)
	states_split = state_str.decode().split('|')
	x_t = [float(states_split[0]), float(states_split[1]), float(states_split[2]), float(states_split[3]), float(states_split[4]), float(states_split[5]), float(states_split[6]), float(states_split[7]), float(states_split[8]), float(states_split[9]), float(states_split[10]), float(states_split[11]), float(states_split[12])]
	x_t = np.array(x_t).reshape(n,1)

	#Step the model one timestep to account for delay caused by solver time
	phi_t = x_t[0]
	theta_t = x_t[1]
	psi_t = x_t[2]
	r_z_left = -x_t[5]
	r_z_right = -x_t[5]

	I_world = np.array([[(Ixx*cos(psi_t) + Iyx*sin(psi_t))*cos(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*sin(psi_t), -(Ixx*cos(psi_t) + Iyx*sin(psi_t))*sin(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*cos(psi_t), Ixz*cos(psi_t) + Iyz*sin(psi_t)], [(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*cos(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*sin(psi_t), -(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*sin(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*cos(psi_t), -Ixz*sin(psi_t) + Iyz*cos(psi_t)], [Ixy*sin(psi_t) + Izx*cos(psi_t), Ixy*cos(psi_t) - Izx*sin(psi_t), Izz]])

	r_left_skew_symmetric = np.array([[0, -r_z_left, r_y_left],
									[r_z_left, 0, -r_x_left],
									[-r_y_left, r_x_left, 0]])

	r_right_skew_symmetric = np.array([[0, -r_z_right, r_y_right],
									[r_z_right, 0, -r_x_right],
									[-r_y_right, r_x_right, 0]])

	A_c = np.array([[0, 0, 0, 0, 0, 0, cos(psi_t)*cos(theta_t), sin(phi_t)*sin(theta_t)*cos(psi_t) - sin(psi_t)*cos(phi_t), sin(phi_t)*sin(psi_t) + sin(theta_t)*cos(phi_t)*cos(psi_t), 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, sin(psi_t)*cos(theta_t), sin(phi_t)*sin(psi_t)*sin(theta_t) + cos(phi_t)*cos(psi_t), -sin(phi_t)*cos(psi_t) + sin(psi_t)*sin(theta_t)*cos(phi_t), 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, -sin(theta_t), sin(phi_t)*cos(theta_t), cos(phi_t)*cos(theta_t), 0, 0, 0, 0],
					
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
					[np.linalg.inv(I_world) @ r_left_skew_symmetric, np.linalg.inv(I_world) @ r_right_skew_symmetric],
					[1/m_value, 0, 0, 1/m_value, 0, 0],
					[0, 1/m_value, 0, 0, 1/m_value, 0],
					[0, 0, 1/m_value, 0, 0, 1/m_value],
					[0, 0, 0, 0, 0, 0]])

	A_d, B_d = discretize_ss(A_c, B_c, dt)

	x_t_temp = A_d @ np.array(x_t).reshape(n,1) + B_d @ control_history[-2]

	phi_t = x_t_temp[0]
	theta_t = x_t_temp[1]
	psi_t = x_t_temp[2]
	r_z_left = -x_t_temp[5]
	r_z_right = -x_t_temp[5]

	I_world = np.array([[(Ixx*cos(psi_t) + Iyx*sin(psi_t))*cos(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*sin(psi_t), -(Ixx*cos(psi_t) + Iyx*sin(psi_t))*sin(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*cos(psi_t), Ixz*cos(psi_t) + Iyz*sin(psi_t)], [(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*cos(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*sin(psi_t), -(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*sin(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*cos(psi_t), -Ixz*sin(psi_t) + Iyz*cos(psi_t)], [Ixy*sin(psi_t) + Izx*cos(psi_t), Ixy*cos(psi_t) - Izx*sin(psi_t), Izz]])

	r_left_skew_symmetric = np.array([[0, -r_z_left, r_y_left],
									[r_z_left, 0, -r_x_left],
									[-r_y_left, r_x_left, 0]])

	r_right_skew_symmetric = np.array([[0, -r_z_right, r_y_right],
									[r_z_right, 0, -r_x_right],
									[-r_y_right, r_x_right, 0]])

	A_c = np.array([[0, 0, 0, 0, 0, 0, cos(psi_t)*cos(theta_t), sin(phi_t)*sin(theta_t)*cos(psi_t) - sin(psi_t)*cos(phi_t), sin(phi_t)*sin(psi_t) + sin(theta_t)*cos(phi_t)*cos(psi_t), 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, sin(psi_t)*cos(theta_t), sin(phi_t)*sin(psi_t)*sin(theta_t) + cos(phi_t)*cos(psi_t), -sin(phi_t)*cos(psi_t) + sin(psi_t)*sin(theta_t)*cos(phi_t), 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, -sin(theta_t), sin(phi_t)*cos(theta_t), cos(phi_t)*cos(theta_t), 0, 0, 0, 0],
					
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
					[np.linalg.inv(I_world) @ r_left_skew_symmetric, np.linalg.inv(I_world) @ r_right_skew_symmetric],
					[1/m_value, 0, 0, 1/m_value, 0, 0],
					[0, 1/m_value, 0, 0, 1/m_value, 0],
					[0, 0, 1/m_value, 0, 0, 1/m_value],
					[0, 0, 0, 0, 0, 0]])

	A_d, B_d = discretize_ss(A_c, B_c, dt)

	x_t_temp = A_d @ np.array(x_t_temp).reshape(n,1) + B_d @ control_history[-1]

	P_param[:, 0] = x_t_temp.reshape(n).copy()
	#P_param[:, 0] = x_t.reshape(n).copy()

	setup_start_time = time.time()

	if iterations % contact_swap_interval == 0:
		swing_left = not swing_left
		swing_right = not swing_right

	swing_left_temp = swing_left
	swing_right_temp = swing_right

	for k in range(N):

		if (iterations+k) % contact_swap_interval == 0 and k is not 0:
			swing_left_temp = not swing_left_temp
			swing_right_temp = not swing_right_temp
			
		D_current_temp = np.array([[int(swing_left_temp == True), 0, 0, 0, 0, 0],
									[0, int(swing_left_temp == True), 0, 0, 0, 0],
									[0, 0, int(swing_left_temp == True), 0, 0, 0],
									[0, 0, 0, int(swing_right_temp == True), 0, 0],
									[0, 0, 0, 0, int(swing_right_temp == True), 0],
									[0, 0, 0, 0, 0, int(swing_right_temp == True)]]) # Swing = no contact (and thus 1 any force == 0 must mean force is 0)    
		
		#P_param[:m, 1+N+n*N+m*N+(k*m):1+N+n*N+m*N+(k*m)+m] = D_current_temp.copy()

		#print("D_k in P_Param:\n", P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m])
		
		# if P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][0, 0] == 1 and P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][3, 3] == 1: # No feet in contact
		# 	P_param[m:m+m, 1 + N + n*N + m*N + (k*m)] = np.array([0, 0, 0, 0, 0, 0]).copy()
		# 	#print("No feet in contact")
		# elif P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][0, 0] == 1 and P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][3, 3] == 0: # Right foot in contact
		# 	P_param[m:m+m, 1 + N + n*N + m*N + (k*m)] = np.array([0, 0, 0, 0, 0, m_value * 9.81]).copy()
		# 	#print("Right foot in contact")
		# elif  P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][0, 0] == 0 and P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][3, 3] == 1: # Left foot in contact
		# 	P_param[m:m+m, 1 + N + n*N + m*N + (k*m)] = np.array([0, 0, m_value * 9.81, 0, 0, 0]).copy()
		# 	#print("Left foot in contact")
		# if  P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][0, 0] == 0 and P_param[:m, 1 + N + n*N + m*N + k*m: 1 + N + n*N + m*N + k*m + m][3, 3] == 0: # Both feet in contact
		# 	P_param[m:m+m, 1 + N + n*N + m*N + (k*m)] = np.array([0, 0, (m_value * 9.81) / 2, 0, 0, (m_value * 9.81) / 2]).copy()

		P_param[m:m+m, 1 + N + n*N + m*N + (k*m)] = np.array([0, 0, (m_value * 9.81) / 2, 0, 0, (m_value * 9.81) / 2]).copy()

	#print("D_k:\n", P_param[:m, 1 + N + n*N + m*N: 1 + N + n*N + m*N + m])

	if foot_behind_left:
		r_y_left = -step_length
	else:
		r_y_left = step_length
			
	if foot_behind_right:
		r_y_right = -step_length
	else:
		r_y_right = step_length

	if iterations % (contact_swap_interval * 2) == 0:
		foot_behind_left = not foot_behind_left
		foot_behind_right = not foot_behind_right

	r_z_left = -x_t[5]
	r_z_right = -x_t[5]

	x_ref = np.zeros((n, N))

	pos_y_temp = pos_y_desired
	psi_temp = psi_desired

	for i in range(N):
		if t > 3:
			pos_y_temp += vel_y_desired * dt
			psi_temp += omega_z_desired * dt
		x_ref[:, i] = np.array([0, 0, psi_temp, 0, pos_y_temp, pos_z_desired, 0, 0, omega_z_desired, 0, vel_y_desired, vel_z_desired, -9.81]).reshape(n)
	
	if t > 3:
		pos_y_desired += vel_y_desired * dt
		psi_desired += omega_z_desired * dt

	P_param[:, 1:1+N] = x_ref.copy()

	foot_behind_left_temp = foot_behind_left
	foot_behind_right_temp = foot_behind_right

	r_y_left_prev = r_y_left
	r_y_right_prev = r_y_right

	for i in range(N):
		if i < N-1:
			phi_t = X_t[n*(i+1)+0, 0]
			theta_t = X_t[n*(i+1)+1, 0]
			psi_t = X_t[n*(i+1)+2, 0]
			
			vel_x_t = X_t[n*(i+1)+9, 0]
			vel_y_t = X_t[n*(i+1)+10, 0]
			vel_z_t = X_t[n*(i+1)+11, 0]
			
			pos_x_t = X_t[n*(i+1)+3, 0]
			pos_y_t = X_t[n*(i+1)+4, 0]
			pos_y_t_next = X_t[n*(i+2)+4, 0]
			pos_z_t = X_t[n*(i+1)+5, 0]
		else:
			phi_t = X_t[n*(N-1), 0]
			theta_t = X_t[n*(N-1)+1, 0]
			psi_t = X_t[n*(N-1)+2, 0]
			
			vel_x_t = X_t[n*(N-1)+9, 0]
			vel_y_t = X_t[n*(N-1)+10, 0]
			vel_z_t = X_t[n*(N-1)+11, 0]
			
			pos_x_t = X_t[n*(N-1)+3, 0]
			pos_y_t = X_t[n*(N-1)+4, 0]
			pos_y_t_next = X_t[n*N + 4, 0]
			pos_z_t = X_t[n*(N-1)+5, 0]
			
		if i == 0:
			phi_t = x_t[0]
			theta_t = x_t[1]
			psi_t = x_t[2]

			vel_x_t = x_t[9]
			vel_y_t = x_t[10]
			vel_z_t = x_t[11]

			pos_x_t = x_t[3]
			pos_y_t = x_t[4]
			pos_z_t = x_t[5]
			
		if foot_behind_left_temp:
			r_y_left = -step_length
		else:
			r_y_left = step_length
			
		if foot_behind_right_temp:
			r_y_right = -step_length
		else:
			r_y_right = step_length
		
		if (iterations+i) % (contact_swap_interval * 2) == 0 and i is not 0:
			foot_behind_left_temp = not foot_behind_left_temp
			foot_behind_right_temp = not foot_behind_right_temp
		
		r_z_left = -pos_z_t
		r_z_right = -pos_z_t
		
		#r_y_left -= (pos_y_t_next - pos_y_t)
		#r_y_right -= (pos_y_t_next - pos_y_t)

		I_world = np.array([[(Ixx*cos(psi_t) + Iyx*sin(psi_t))*cos(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*sin(psi_t), -(Ixx*cos(psi_t) + Iyx*sin(psi_t))*sin(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*cos(psi_t), Ixz*cos(psi_t) + Iyz*sin(psi_t)], [(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*cos(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*sin(psi_t), -(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*sin(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*cos(psi_t), -Ixz*sin(psi_t) + Iyz*cos(psi_t)], [Ixy*sin(psi_t) + Izx*cos(psi_t), Ixy*cos(psi_t) - Izx*sin(psi_t), Izz]])
		
		# Skew symmetric versions for the 3x1 foot position vector resembling the matrix version of the cross product of two vectors. This is needed for the matrix form.
		r_left_skew_symmetric = np.array([[0, -r_z_left, r_y_left],
											[r_z_left, 0, -r_x_left],
											[-r_y_left, r_x_left, 0]])

		r_right_skew_symmetric = np.array([[0, -r_z_right, r_y_right],
											[r_z_right, 0, -r_x_right],
											[-r_y_right, r_x_right, 0]])

		A_c = np.array([[0, 0, 0, 0, 0, 0, cos(psi_t)*cos(theta_t), sin(phi_t)*sin(theta_t)*cos(psi_t) - sin(psi_t)*cos(phi_t), sin(phi_t)*sin(psi_t) + sin(theta_t)*cos(phi_t)*cos(psi_t), 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, sin(psi_t)*cos(theta_t), sin(phi_t)*sin(psi_t)*sin(theta_t) + cos(phi_t)*cos(psi_t), -sin(phi_t)*cos(psi_t) + sin(psi_t)*sin(theta_t)*cos(phi_t), 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, -sin(theta_t), sin(phi_t)*cos(theta_t), cos(phi_t)*cos(theta_t), 0, 0, 0, 0],
						
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
						[np.linalg.inv(I_world) @ r_left_skew_symmetric, np.linalg.inv(I_world) @ r_right_skew_symmetric],
						[1/m_value, 0, 0, 1/m_value, 0, 0],
						[0, 1/m_value, 0, 0, 1/m_value, 0],
						[0, 0, 1/m_value, 0, 0, 1/m_value],
						[0, 0, 0, 0, 0, 0]])
		
		A_d, B_d = discretize_ss(A_c, B_c, dt)
		
		P_param[:, 1 + N + (i*n):1 + N + (i*n)+n] = A_d.copy()
		P_param[:, 1 + N + n * N + (i*m):1 + N + n * N + (i*m)+m] = B_d.copy()

	r_y_left = r_y_left_prev
	r_y_right = r_y_right_prev

	setup_end_time = time.time()
	
	x0_solver = vertcat(*[X_t, U_t])

	sol = solver(x0=x0_solver, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=P_param)

	u_t = sol['x'][n * (N+1) : n * (N+1) + m]
	control_history.append(np.array(u_t).reshape(m,1).copy())

	msg = "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}".format(u_t[0], u_t[1], u_t[2], u_t[3], u_t[4], u_t[5], r_x_left, r_y_left, r_z_left, r_x_right, r_y_right, r_z_right, x_t_temp[1][0])
	print("msg:", msg)

	mpc_socket.sendto(bytes(msg, "utf-8"), mpc_addr)

	if legs_attached:

		#mpc_socket.sendto(bytes("0|0|0|0|0|0|0|0|0|0|0|0", "utf-8"), addr)

		left_leg_state_str, left_leg_addr = left_leg_socket.recvfrom(4096)
		#left_leg_state_str = b"0|0|0|0|0|0|0|0|0|0"
		left_leg_state_split = left_leg_state_str.decode().split('|')

		right_leg_state_str, right_leg_addr = right_leg_socket.recvfrom(4096)
		#right_leg_state_str = b"0|0|0|0|0|0|0|0|0|0"
		right_leg_state_split = right_leg_state_str.decode().split('|')

		min_torque = -150
		max_torque = 150

		if t > 0.05:
			left_leg_torques = get_joint_torques(-u_t[0], -u_t[1], -u_t[2], float(left_leg_state_split[0]), float(left_leg_state_split[1]), float(left_leg_state_split[2]), float(left_leg_state_split[3]), float(left_leg_state_split[4]), x_t[0], x_t[1], x_t[2])
			left_leg_msg = f"{left_leg_torques[0, 0]}|{left_leg_torques[1, 0]}|{left_leg_torques[2, 0]}|{left_leg_torques[3, 0]}|{left_leg_torques[4, 0]}"
			
			right_leg_torques = get_joint_torques(-u_t[3], -u_t[4], -u_t[5], float(right_leg_state_split[0]), float(right_leg_state_split[1]), float(right_leg_state_split[2]), float(right_leg_state_split[3]), float(right_leg_state_split[4]), x_t[0], x_t[1], x_t[2])
			right_leg_msg = f"{right_leg_torques[0, 0]}|{right_leg_torques[1, 0]}|{right_leg_torques[2, 0]}|{right_leg_torques[3, 0]}|{right_leg_torques[4, 0]}"

			for i in range(5):
				if left_leg_torques[i] > max_torque:
					left_leg_torques[i] = max_torque
				elif left_leg_torques[i] < min_torque:
					left_leg_torques[i] = min_torque

				if right_leg_torques[i] > max_torque:
					right_leg_torques[i] = max_torque
				elif right_leg_torques[i] < min_torque:
					right_leg_torques[i] = min_torque

			print("left_leg_msg:", left_leg_msg)
			print("right_leg_msg:", right_leg_msg)
		else:
			left_leg_msg = "0|0|0|0|0"
			right_leg_msg = "0|0|0|0|0"

			left_leg_torques = np.array([[0], [0], [0], [0], [0]])
			right_leg_torques = np.array([[0], [0], [0], [0], [0]])
		
		left_leg_socket.sendto(bytes(left_leg_msg, "utf-8"), left_leg_addr)
		right_leg_socket.sendto(bytes(right_leg_msg, "utf-8"), right_leg_addr)

		log_file.write(f"{t},{x_t[0]},{x_t[1]},{x_t[2]},{x_t[3]},{x_t[4]},{x_t[5]},{x_t[6]},{x_t[7]},{x_t[8]},{x_t[9]},{x_t[10]},{x_t[11]},{x_t[12]}," + msg.replace("|", ",") + f",{left_leg_torques[0, 0]},{left_leg_torques[1, 0]},{left_leg_torques[2, 0]},{left_leg_torques[3, 0]},{left_leg_torques[4, 0]},{right_leg_torques[0, 0]},{right_leg_torques[1, 0]},{right_leg_torques[2, 0]},{right_leg_torques[3, 0]},{right_leg_torques[4, 0]},{float(left_leg_state_split[0])},{float(left_leg_state_split[1])},{float(left_leg_state_split[2])},{float(left_leg_state_split[3])},{float(left_leg_state_split[4])},{float(left_leg_state_split[5])},{float(left_leg_state_split[6])},{float(left_leg_state_split[7])},{float(left_leg_state_split[8])},{float(left_leg_state_split[9])},{float(right_leg_state_split[0])},{float(right_leg_state_split[1])},{float(right_leg_state_split[2])},{float(right_leg_state_split[3])},{float(right_leg_state_split[4])},{float(right_leg_state_split[5])},{float(right_leg_state_split[6])},{float(right_leg_state_split[7])},{float(right_leg_state_split[8])},{float(right_leg_state_split[9])}" + "\n")
	else:
		log_file.write(f"{t},{x_t[0]},{x_t[1]},{x_t[2]},{x_t[3]},{x_t[4]},{x_t[5]},{x_t[6]},{x_t[7]},{x_t[8]},{x_t[9]},{x_t[10]},{x_t[11]},{x_t[12]}," + msg.replace("|", ",") + f",{x_t_temp[1][0]}"+ f",{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}" + "\n")

	X_t[:-n] = sol['x'][n:n * (N+1)]
	X_t[-n:] = sol['x'][-n - m * N: n * (N+1)]

	U_t[:-m] = sol['x'][m + n * (N+1):]
	U_t[-m:] = sol['x'][-m:]

	#print("Optimal state sequence:", sol['x'][:n*(N+1)])
	#print("Optimal control sequence:", sol['x'][n * (N+1):])

	if t > 5 and False:
		
		np.set_printoptions(threshold=sys.maxsize)
		#print("Full P_param:\n", np.array(P_param))
		np.save("/tmp/P_param.npy", P_param)
		
		print("x_t before stepping:", x_t)

		phi_t = x_t[0]
		theta_t = x_t[1]
		psi_t = x_t[2]
		
		I_world = np.array([[(Ixx*cos(psi_t) + Iyx*sin(psi_t))*cos(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*sin(psi_t), -(Ixx*cos(psi_t) + Iyx*sin(psi_t))*sin(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*cos(psi_t), Ixz*cos(psi_t) + Iyz*sin(psi_t)], [(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*cos(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*sin(psi_t), -(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*sin(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*cos(psi_t), -Ixz*sin(psi_t) + Iyz*cos(psi_t)], [Ixy*sin(psi_t) + Izx*cos(psi_t), Ixy*cos(psi_t) - Izx*sin(psi_t), Izz]])
		
		r_z_left = -x_t[5]
		r_z_right = -x_t[5]
		
		r_left_skew_symmetric = np.array([[0, -r_z_left, r_y_left],
										[r_z_left, 0, -r_x_left],
										[-r_y_left, r_x_left, 0]])
		
		r_right_skew_symmetric = np.array([[0, -r_z_right, r_y_right],
										[r_z_right, 0, -r_x_right],
										[-r_y_right, r_x_right, 0]])
		
		A_c = np.array([[0, 0, 0, 0, 0, 0, cos(psi_t)*cos(theta_t), sin(phi_t)*sin(theta_t)*cos(psi_t) - sin(psi_t)*cos(phi_t), sin(phi_t)*sin(psi_t) + sin(theta_t)*cos(phi_t)*cos(psi_t), 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, sin(psi_t)*cos(theta_t), sin(phi_t)*sin(psi_t)*sin(theta_t) + cos(phi_t)*cos(psi_t), -sin(phi_t)*cos(psi_t) + sin(psi_t)*sin(theta_t)*cos(phi_t), 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, -sin(theta_t), sin(phi_t)*cos(theta_t), cos(phi_t)*cos(theta_t), 0, 0, 0, 0],
						
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
						[np.linalg.inv(I_world) @ r_left_skew_symmetric, np.linalg.inv(I_world) @ r_right_skew_symmetric],
						[1/m_value, 0, 0, 1/m_value, 0, 0],
						[0, 1/m_value, 0, 0, 1/m_value, 0],
						[0, 0, 1/m_value, 0, 0, 1/m_value],
						[0, 0, 0, 0, 0, 0]])
		
		A_d, B_d = discretize_ss(A_c, B_c, dt)
		
		x_t = A_d @ np.array(x_t).reshape(n,1) + B_d @ control_history[-2]

		for i in range(N-1):
			phi_t = x_t[0]
			theta_t = x_t[1]
			psi_t = x_t[2]
			
			I_world = np.array([[(Ixx*cos(psi_t) + Iyx*sin(psi_t))*cos(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*sin(psi_t), -(Ixx*cos(psi_t) + Iyx*sin(psi_t))*sin(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*cos(psi_t), Ixz*cos(psi_t) + Iyz*sin(psi_t)], [(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*cos(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*sin(psi_t), -(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*sin(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*cos(psi_t), -Ixz*sin(psi_t) + Iyz*cos(psi_t)], [Ixy*sin(psi_t) + Izx*cos(psi_t), Ixy*cos(psi_t) - Izx*sin(psi_t), Izz]])

			r_z_left = -x_t[5]
			r_z_right = -x_t[5]
			
			r_left_skew_symmetric = np.array([[0, -r_z_left, r_y_left],
											[r_z_left, 0, -r_x_left],
											[-r_y_left, r_x_left, 0]])
			
			r_right_skew_symmetric = np.array([[0, -r_z_right, r_y_right],
											[r_z_right, 0, -r_x_right],
											[-r_y_right, r_x_right, 0]])
			
			A_c = np.array([[0, 0, 0, 0, 0, 0, cos(psi_t)*cos(theta_t), sin(phi_t)*sin(theta_t)*cos(psi_t) - sin(psi_t)*cos(phi_t), sin(phi_t)*sin(psi_t) + sin(theta_t)*cos(phi_t)*cos(psi_t), 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, sin(psi_t)*cos(theta_t), sin(phi_t)*sin(psi_t)*sin(theta_t) + cos(phi_t)*cos(psi_t), -sin(phi_t)*cos(psi_t) + sin(psi_t)*sin(theta_t)*cos(phi_t), 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, -sin(theta_t), sin(phi_t)*cos(theta_t), cos(phi_t)*cos(theta_t), 0, 0, 0, 0],
							
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
							[np.linalg.inv(I_world) @ r_left_skew_symmetric, np.linalg.inv(I_world) @ r_right_skew_symmetric],
							[1/m_value, 0, 0, 1/m_value, 0, 0],
							[0, 1/m_value, 0, 0, 1/m_value, 0],
							[0, 0, 1/m_value, 0, 0, 1/m_value],
							[0, 0, 0, 0, 0, 0]])
			
			A_d, B_d = discretize_ss(A_c, B_c, dt)
			
			x_t = A_d @ np.array(x_t).reshape(n,1) + B_d @ sol['x'][n * (N+1) + (i*m) + m:n * (N+1) + (i*m) + m + m]

			print("x_t at i=",i,"\n",x_t)

		while True:
			test = 0

	t += dt
	iterations += 1

	end_time = time.time()
	
	print("Loop frequency:", 1/(end_time - start_time), "Hz")
	#print("Rest of the iteration (without setup):", (end_time - start_time) - (setup_end_time - setup_start_time))
	remainder = 0
	if 1/30 - (end_time - start_time) > 0:
		remainder = 1/30 - (end_time - start_time)
		time.sleep(remainder)