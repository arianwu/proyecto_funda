import numpy as np
from copy import copy
import rbdl

pi = np.pi
cos = np.cos
sin = np.sin

def Qinv(Q):
    Q = Q.copy()
    Q[1:4] = -Q[1:4]
    return Q

def Qskew(Q):
    w, ex, ey, ez = Q
    return np.array([[ w, -ex, -ey, -ez],
                     [ex,   w, -ez,  ey],
                     [ey,  ez,   w, -ex],
                     [ez, -ey,  ex,   w]])

def Qdot(Q1,Q2):
    return np.dot(Qskew(Q1), Q2)

def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R

def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T

def fkine(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]

    """
    # Longitudes (en metros)
    lh = 0.05
    lb = 0.29
    la = 0.26
    lm = 0.05
    ld = 0.07
    # Matrices DH
    Tpre = dh(0, 0, 0, pi/2)
    T1 = dh(lh , q[0]-pi/2, 0 , pi/2)
    T2 = dh(0  , q[1]+pi/2, 0 , pi/2)
    T3 = dh(lb , q[2]+pi/2, 0 , pi/2)
    T4 = dh(0  , q[3]+pi  , 0 , pi/2)
    T5 = dh(la , q[4]+pi  , 0 , pi/2)
    T6 = dh(0  , q[5]+pi/2, lm, pi/2)
    T7 = dh(0  , q[6]+pi/2, 0 , pi/2)
    Tpost = dh(ld, 0, 0, 0)
    # Efector final con respecto a la base
    T = Tpre.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7).dot(Tpost)
    return T

def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Alocacion de memoria
    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine(q)
    # Iteracion para la derivada de cada columna
    for i in range(7):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] += delta
        # Transformacion homogenea luego del incremento (q+dq)
        dT = fkine(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[:,i] = ((dT-T)[0:3,3])
    return J/delta

def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Alocacion de memoria
    J = np.zeros((7,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine(q)
    x0 = TF2xyzquat(T)
    # Implementar este Jacobiano aqui
    for i in range(7):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] += delta
        # Transformacion homogenea luego del incremento (q+dq)
        dT = fkine(dq)
        dx = TF2xyzquat(dT)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J.T[i,:] = (dx-x0)
    
    return J/delta

def ikine_newton_position(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo de newton
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_position(q, delta)
        e = xdes - fkine(q)[0:3,3]
        q = q + np.linalg.pinv(J).dot(e)
        if(np.linalg.norm(e)<epsilon):
            break
    return (q+pi)%(2*pi)-pi

    
def ikine_gradient_position(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo gradiente
    """
    epsilon  = 0.001
    max_iter = 150
    delta    = 0.00001
    alpha = 0.5
    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_position(q, delta)
        e = xdes - fkine(q)[0:3,3]
        q = q + alpha*J.T.dot(e)
        if(np.linalg.norm(e)<epsilon):
            break
    return (q+pi)%(2*pi)-pi

def ikine_position(xdes, q0):
    q1 = ikine_gradient_position(xdes, q0)
    q2 = ikine_newton_position(xdes, q1)
    return q2

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/robot_gazebo.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        try:
            ddq = np.linalg.inv(self.M).dot(tau-self.b)
        except np.linalg.LinAlgError:
            ddq = self.M.T.dot( np.linalg.inv( np.dot(self.M,self.M.T)+0.9**2*np.eye(self.q.size) ) ).dot(tau-self.b)
        
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq