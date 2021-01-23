#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from roslib import packages

import rbdl


rospy.init_node("control_dininv")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual  = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])
# Archivos donde se almacenara los datos
fqact = open("/home/user/proyecto_ws/src/proy/graphs/qactual.txt", "w")
fqdes = open("/home/user/proyecto_ws/src/proy/graphs/qdeseado.txt", "w")
fxact = open("/home/user/proyecto_ws/src/proy/graphs/xactual.txt", "w")
fxdes = open("/home/user/proyecto_ws/src/proy/graphs/xdeseado.txt", "w")

# Nombres de las articulaciones
jnames = ["base_joint", "hombro_joint", "antebrazo_joint", "ante_brazo_joint", "brazo_joint", "brazo_mano_joint", "mano_joint"]
# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q = np.array([0.0, 1.0, 0.5, 1.8, -0.7, 0.0, 0.0])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Aceleracion inicial
ddq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Configuracion articular deseada
qdes = np.array([-0.732, 1.101, 0.13, -1.896, -1.589, 0.134, 0.])
# Velocidad articular deseada
dqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Aceleracion articular deseada
ddqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine(qdes)[0:3,3]
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel('../urdf/robot_gazebo.urdf')
ndof   = modelo.q_size 	# Grados de libertad
zeros = np.zeros(ndof) 	# Vector de ceros

# Frecuencia del envio (en Hz)
freq = 20
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Bucle de ejecucion continua
t = 0.0

# Se definen las ganancias del controlador
valores = 1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

while not rospy.is_shutdown():

	# Leer valores del simulador
	q  = robot.read_joint_positions()
	dq = robot.read_joint_velocities()
	# Posicion actual del efector final
	x = fkine(q)[0:3,3]
	# Tiempo actual (necesario como indicador para ROS)
	jstate.header.stamp = rospy.Time.now()

	# Almacenamiento de datos
	fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
	fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+
            	str(xdes[2])+'\n')
	fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+
            	' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+' '+str(q[6])+'\n ')
	fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+
            	' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+' '+str(qdes[6])+'\n ')

	# ----------------------------
	# Control dinamico (COMPLETAR)
	# ----------------------------
	y = ddqdes + Kd.dot(dqdes-dq) + Kp.dot(qdes-q)
    

	M2 = np.zeros([ndof, ndof])  # Para matriz de inercia
	b2 = np.zeros(ndof)      	# Para efectos no lineales
	rbdl.NonlinearEffects(modelo, q, dq, b2)
	rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)

	u = M2.dot(y) + b2
	#print(u)

	# Simulacion del robot
	robot.send_command(u)

	# Publicacion del mensaje
	jstate.position = q
	pub.publish(jstate)
	bmarker_deseado.xyz(xdes)
	bmarker_actual.xyz(x)
	t = t+dt
	# Esperar hasta la siguiente  iteracion
	rate.sleep()

fqact.close()
fqdes.close()
fxact.close()
fxdes.close()
