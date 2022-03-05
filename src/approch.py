#!/usr/bin/env python3
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
        self.outx = self.pidx.update(self.rel_posx, self.rel_velx)
        self.outz = self.pidz.update(self.rel_posz, self.rel_velz)

        




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


