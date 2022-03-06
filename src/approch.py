#!/usr/bin/env python3
from mmap import ACCESS_DEFAULT
import controler
from time import sleep
import numpy as np
import math
import matplotlib.pyplot as plt
import PID

'lol just checking git'
def next_move(xyz, angle, clockwise = True):
    
    if clockwise:
        if angle > 5:
            r = math.sqrt(np.sum(np.array(xyz)*np.array(xyz)))
            dx = -1*r*(math.cos(math.radians(angle - 5)) - math.cos(math.radians(angle)))
            dy = -1*r*(math.sin(math.radians(angle - 5)) - math.sin(math.radians(angle)))
            return dx, dy
        else:
            return 0,0

    else:
        if angle > 5:
            r = math.sqrt(np.sum(np.array(xyz)*np.array(xyz)))
            dx = r*(math.cos(math.radians(angle - 5)) - math.cos(math.radians(angle)))
            dy = r*(math.sin(math.radians(angle - 5)) - math.sin(math.radians(angle)))
            return dx, dy
        else:
            return 0,0

def d_yaw_cal(xyz):
    return math.degrees(math.atan2(xyz[2], xyz[0])) - 90




class follow:
    def __init__(self, xyz0) -> None:
        self.pidx = PID.PID()
        self.pidz = PID.PID()
        self.frequency = 20
        self.xyz0 = xyz0

    def relative_vel(self):
        self.rel_posx = self.xyz1[0]
        self.rel_posz = self.xyz1[2]
        self.rel_velx = (self.xyz1[0] - self.xyz0[0])*self.frequency
        self.rel_velz = (self.xyz1[2] - self.xyz0[2])*self.frequency

    def update(self, xyz1):
        self.xyz1 = xyz1
        self.relative_vel()
        self.xyz0 = self.xyz1
        self.outx = self.pidx.update(self.rel_posx, self.rel_velx) # acceleration in x and z
        self.outz = self.pidz.update(self.rel_posz, self.rel_velz)


# def clear_rotation_offset():

def camera_to_local(x,y,z):
    return z, -x,-y 

def local_to_camera(x,y,z):
    return -y, -z, x 



if __name__ == '__main__':
    ic = controler.Flight_controller(cv= True)
    sleep(2)
    print(ic.curr_x)
    ic.set_offboard_mode()
    while ic.mast_angle > 5:
        if ic.mast_sense > 0:
            clockwise = True
        else:
            clockwise = False
        print('mast angle = '+str(ic.mast_angle))
        dx, dy = next_move(ic.mast_xyz, ic.mast_angle, clockwise= clockwise)
        dyaw = d_yaw_cal(ic.mast_xyz)
        
        ic.d_yaw(dyaw=dyaw)
        ic.move_wrtDrone(dx, dy, 0)
        ic.set_pose()
    
    foll = follow(ic.mast_xyz)
    while True:
        foll.update(ic.mast_xyz)
        accel_x = foll.outx # left and right
        # accel_z = foll.outz
        accel_z = 0 # in and out
        accel_y = 0 # up and down
        yaw = -1*math.radians(ic.curr_yaw) # get the current yaw of the drone
        rotation_matrix = np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]]) # define the rotation matrix
        accel_x,accel_y,accel_z = camera_to_local(accel_x,accel_y,accel_z) # get the acceleration of the drone in the real non-offset global frame
        accel_old = np.array([accel_x,accel_y])
        accel_new = np.matmul(rotation_matrix,accel_old)
        ic.accel_command(accel_new[0],accel_new[1], accel_z)







