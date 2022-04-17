from py4j.java_gateway import JavaGateway
import tkinter
from tkinter import messagebox
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

'''
Update the environment state based on action chosen
0 = downscale
1 = upscale
2 = do nothing
'''

class ImaginationEnvironment(gym.Env):
    power = 0
    max_power = 0
    idle_power = 0
    count = -1
    host_pe = 0
    vm_pe = 0
    vm = 0
    host = 0
    value = 0
    left = 0
    currentTime = 0.0
    currentID = 0
    maxSLA = 0.0
    minSLA = 0.0
    
    def __init__(self, value):
        super(ImaginationEnvironment, self).__init__()
        
        self.done = False 
        self.seed()
        self.reward = 0
        self.action_space = 3
        self.state_space = 9
        self.value = value
        self.count = 0

    def getSimulation(self, simulationGateway):
        #gateway = JavaGateway(start_callback_server=True)
        #self.simulation = gateway.entry_point

        try:
            self.simulation = simulationGateway
        
            self.max_power = self.simulation.getMaxPower();
            self.idle_power = self.simulation.getIdlePower();
            self.vm_pe = self.simulation.getVMPENumber();
            self.host_pe = self.simulation.getHostPENumber();
            self.left = self.simulation.getCloudletLeft();
            self.currentTime = self.simulation.getCurrentTime();
        
            self.vm = self.simulation.getVMNumber();
            self.host = self.simulation.getHostNumber();

            self.maxSLA = self.max_power * self.simulation.getMaxSLA();
            self.minSLA = self.max_power * self.simulation.getMinSLA();
        except:
            messagebox.showerror("Error", "Cloud Sim Plus is not started")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #self.count = self.count + 1 ;
        self.currentId = self.simulation.getCurrentID()
        self.currentTime = self.simulation.getCurrentTime()
        self.power = self.simulation.getPowerUsage()
        self.reward = 0;

        #print("Power : ",self.power)
        #print ("Prev : ",self.energy[self.count]," : ")
        if action == 0 :
            print ("Scale down")
            self.power = self.scaleDown()
            #self.scaleDown();
        if action == 1 :
            print ("Scale up")
            self.power = self.scaleUp()
            #self.scaleUp();
        if action == 2 :
            print ("Do nothing")

        #print ("Post : ",self.energy[self.count]," : ")
        
        # Calculate the "reward" for the new state
        self.reward = self.get_reward();
        
        #print(self.reward)
        
        # Store the new "observation" for the state
        state = self.get_state()
        
        # Return the state, reward and done
        return state, self.reward, self.done, {}

    def scaleDown(self):
        #Bagi Compltee
        #self.power = self.power * 0.8
        try:
            temp = self.simulation.scaleDown(self.currentId)
        except:
            temp = 0

        return temp

    def scaleUp(self):
        #self.power = self.power * 1.2
        try:
            temp = self.simulation.scaleUp(self.currentId)
        except:
            temp = 0

        return temp

    def reset(self):
        self.done = False
        self.count = -1;
        #self.observation = np.zeros([9, 1])
        #return self.observation
        state = self.get_state();
        return state;

    def get_reward(self):
        mod = 0
        pue = self.power / self.max_power

        maxthreshold = self.max_power * 0.8
        minthreshold = self.max_power * 0.4

        #Basic Premises
        if self.power >= self.max_power :
            mod = -2
        elif self.power > maxthreshold :
            mod = -1
        elif self.power <= maxthreshold and self.power >= minthreshold:
            mod = 1
        elif self.power < minthreshold :
            mod = -1

        #Additional
        if self.left <= 1000:       
            mod += 2
        if self.power > self.maxSLA or self.power < self.minSLA:
            mod -= 2
        if self.power == 0:
            mod -= mod

        return (self.value - pue) * mod;

    def get_state(self):
        self.vm_pe = self.simulation.getVMNumber()
        self.host_pe = self.simulation.getHostNumber()
        self.left = self.simulation.getCloudletLeft()
        
        vm_process_pe = self.vm - self.vm_pe
        host_process_pe = self.host - self.host_pe
        
        state = [int(self.power < self.max_power * 0.4),    #Lower than min threshold
                 int(self.power >= self.max_power * 0.8),   #Higher/equal to max threshold
                 int(self.power < self.max_power * 0.8),    #Lower than max threshold
                 int(self.power >= self.max_power * 0.4),   #Higher than/equal to min threshold
                 int(self.vm_pe > vm_process_pe),           #More free PE than used in VM
                 int(self.host_pe > host_process_pe),       #More free PE than used in Host
                 int(self.left <= 1000),                    #If remaining left is lower than min = 1000
                 int(self.power > self.maxSLA or self.power < self.minSLA), 
                 int(self.power == 0)]                      #Power Usage is zero
        return state;

    def getPower(self):
        return self.power;

    def getTime(self):
        self.currentTime = self.simulation.getCurrentTime()+1;
        return self.currentTime;
    
'''
    NO1
    #getsimulation
    #agent rl
    #make action based on reward
    #put action into simulation
    #scale up/or down

    #NO2
    #run simulation
    #check within range
    #trigger when above/below treshold
    #call agent
''
if __name__ == '__main__':
    ic = ImaginationEnvironment(10)

    while True:
        print(ic.getPower())
'''
