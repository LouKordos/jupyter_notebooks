from casadi import *
import numpy as np

solver = nlpsol("solver", "ipopt", "nlp.so")

opts = {}
opts["print_time"] = 1
#opts["expand"] = False
#opts['ipopt'] = {"max_iter":50, "print_level":3, "acceptable_tol":1e-7, "acceptable_obj_change_tol":1e-5}

solver = nlpsol("solver", "ipopt", "nlp.so", opts)

lbg = []
ubg = []
lbx = []
ubx = []

n = 13
m = 6
N = 30

dt = 1/30

f_min_z = 0
f_max_z = 1000

# Initial state constraints
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
    if i % m == 0:
        lbg += [-inf] * 8
        ubg += [0] * 8

for i in range(n*(N+1)):
    lbx += [-inf]
    ubx += [inf]

for i in range(m*N):
    if i % m == 0:
        lbx += [-inf, -inf, f_min_z, -inf, -inf, f_min_z]
        ubx += [inf, inf, f_max_z, inf, inf, f_max_z]

x_t = [0, 0, 0, 0, 0, 1.48, 0, 0, 0, 0, 0, 0, -9.81]

x_ref = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -9.81]
x_ref = np.tile(np.array(x_ref).reshape(n, 1), N)

U_t = np.zeros((m, N))
X_t = np.zeros((n, N+1))
#X_t = np.tile(np.array(x_t).reshape(n, 1), N+1)

X_t = X_t.reshape((n * (N+1), 1))
U_t = U_t.reshape((m * N, 1))

P_rows = n
P_cols = 1 + N + n * N + m * N + N * m 

P_param = np.zeros((P_rows, P_cols))
P_param[:, 0] = np.array(x_t).reshape(n)
P_param[:, 1:1 + N] = x_ref

swing_left = True
swing_right = False

D = np.array([[int(swing_left == True), 0, 0, 0, 0, 0],
            [0, int(swing_left == True), 0, 0, 0, 0],
            [0, 0, int(swing_left == True), 0, 0, 0],
            [0, 0, 0, int(swing_right == True), 0, 0],
            [0, 0, 0, 0, int(swing_right == True), 0],
            [0, 0, 0, 0, 0, int(swing_right == True)]]) #Swing = no contact (and thus 1 any force == 0 must mean force is 0)

P_param[:m, 1 + N + n*N + m*N:] = np.tile(D, N).copy()

psi_t = 0
theta_t = 0
psi_t = 0

r_x_left = 0.15
r_y_left = 0
r_z_left = -1

r_x_right = -0.15
r_y_right = 0
r_z_right = -1

m_value = 30

I_body = np.array([[0.836, 0., 0.],
                    [0., 1.2288, 0.],
                    [0., 0., 1.4843]]) # Inertia in the body frame (meaning around CoM of torso).

Ixx = I_body[0, 0]
Ixy = I_body[0, 1]
Ixz = I_body[0, 2]

Iyx = I_body[1, 0]
Iyy = I_body[1, 1]
Iyz = I_body[1, 2]

Izx = I_body[2, 0]
Izy = I_body[2, 1]
Izz = I_body[2, 2]

for i in range(N):

    I_world = np.array([[(Ixx*cos(psi_t) + Iyx*sin(psi_t))*cos(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*sin(psi_t), -(Ixx*cos(psi_t) + Iyx*sin(psi_t))*sin(psi_t) + (Ixy*cos(psi_t) + Iyy*sin(psi_t))*cos(psi_t), Ixz*cos(psi_t) + Iyz*sin(psi_t)], [(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*cos(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*sin(psi_t), -(-Ixx*sin(psi_t) + Iyx*cos(psi_t))*sin(psi_t) + (-Ixy*sin(psi_t) + Iyy*cos(psi_t))*cos(psi_t), -Ixz*sin(psi_t) + Iyz*cos(psi_t)], [Ixy*sin(psi_t) + Izx*cos(psi_t), Ixy*cos(psi_t) - Izx*sin(psi_t), Izz]])

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

    P_param[:, 1 + N + (i*n):1 + N + (i*n)+n] = A_d.copy()
    P_param[:, 1 + N + n * N + (i*m):1 + N + n * N + (i*m)+m] = B_d.copy()
    
print("A_d:\n", P_param[:, 1+N:1+N+n], "\n")
print("B_d:\n", P_param[:, 1 + N + n * N:1 + N + n * N+m], "\n")

U_t = np.zeros((m, N))
X_t = np.tile(np.array(x_t).reshape(n, 1), N+1).reshape(n, N+1)

x0_solver = vertcat(*[X_t.reshape((n * (N+1), 1)), U_t.reshape((m * N, 1))])
 
print("lbx length:", len(lbx))
print("ubx length:", len(ubx))
print("lbg length:", len(lbg))
print("ubg length:", len(ubg))

sol = solver(x0=x0_solver, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=P_param)

print(sol['x'][103])
print(sol['x'][420])
print(sol['x'][203])
print(sol['x'][27])
print(sol['x'][522])

print("sol['x'] shape:", sol['x'].shape)