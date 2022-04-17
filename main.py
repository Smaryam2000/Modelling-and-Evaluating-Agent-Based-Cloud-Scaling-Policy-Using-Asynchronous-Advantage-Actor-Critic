import GraphMaker
import tkinter
import tensorflow as tf
from tkinter import ttk;
from tkinter import Button;
from tkinter import Label;
from tkinter import scrolledtext;
from tkinter import messagebox;

from py4j.java_gateway import JavaGateway
from actor_critics import ActorCritic
from environment import ImaginationEnvironment
from Functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from py4j.java_gateway import (JavaGateway, CallbackServerParameters)
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from plot_script import plot_result

######### COPY
import time as t

import nest_asyncio
nest_asyncio.apply()

import asyncio
import threading

''' update global network weight kt sini.
    original weight master 70% & update weight worker 30%
    average global network bahagi 2 (master + worker)
'''
def callback(actor):
    masterWeights = np.array(masterActor.model.get_weights()) * 0.7
    workerWeights = np.array(actor.model.get_weights()) * 0.3
    weights = (masterWeights+workerWeights) / 2
    masterActor.model.set_weights(weights.tolist())
    print("*****************update weights*****************")

'''looping for worker training'''
def wrapper(loop,*args):
    callback(loop.run_until_complete(critic_train(*args)))
############################copy

class RLEDCSS(object):
    start = False;
    ready = False;  #Give Java time to set up
    done = False;
    activated = False;
    action = 0
    sla_violation = 0
    agent_call = 0
    setParam = "1,2,4,50,30,3600"
    def checkConnection(self, msg = "|"):
        #return(msg, " | Check Python")
        return "Said hello from Python called by Java {0}".format(msg)

    def setSimulationStart(self, _start):
        self.start = _start

    def setSimulationDone(self, _done):
        self.done = _done;

    def setSimulationSettings(self, _setParam):
        self.setParam = _setParam

    def setSimulationReady(self, _ready):
        self.ready = _ready

    def setAgentActivated(self, _activated):
        self.activated = _activated

    def setAgentAction(self, action):
        self.action = action

    def setSLAViolationNumber(self, sla_violation):
        self.sla_violation = sla_violation

    def setAgentCallNumber(self, agent_call):
        self.agent_call = agent_call

    def simulationStart(self):
        return self.start;

    def simulationEnd(self):
        return False

    def simulationDone(self):
        return self.done;

    def simulationSettings(self):
        return "{0}".format(self.setParam);

    def agentAction(self):
        return "{0}".format(self.action);

    class Java:
        implements = ['org.cloudsimplus.examples.RLEDCSSInterface']

masterActor = None


async def critic_train(actor, imagineDepth, imaginationEnv, gm, gateway, ax):
    # Actor will be initialized first before training
    # actor = ActorCritic(imaginationEnv, params)   
    global masterActor
    
    if masterActor is None:
        masterActor = ActorCritic(imaginationEnv, params)
    actor = ActorCritic(imaginationEnv, params)

    actor.model = tf.keras.models.clone_model(
        masterActor.model, input_tensors=None, clone_function=None
    )                   
                      
    
    # Initialize CSP connection Enviroment
    imagineEnv.getSimulation(gateway);

    # This variable is used to ping/pong the annotation NO LONGER USED

    # This variable is used to mark the graph
    num_g = 1
    
    sum_of_rewards = []
    sum_of_losses = []

    max_steps = 10000

    #contain the total number of SLA violation for all episode
    sum_sla_violation = 0
    sum_agent_call = 0

    ann_list = []
    #x = np.array([0], dtype = np.float32)
    #y = np.array([0], dtype = np.float32)

    x2 = np.array([0], dtype = np.float32)
    y2 = np.array([0], dtype = np.float32)
    
    #rollout = RolloutStorage(max_steps, imaginationEnv.state_space.shape)

    #Line 11
    for e in range(imagineDepth):
        state = imaginationEnv.reset()
        state = convert_target_to_real_state(state, imaginationEnv.state_space)
        score = 0
        count = 0
        max_steps = 10000

        x = np.array([0], dtype = np.float32)
        y = np.array([0], dtype = np.float32)
        #gm.setPreviousLineToDiff()
        #canvas.set_color('gray')

        '''
            rledcss.done is different from local done
            for now we will use rledcss.done and clean up done latter

            apparantlty local done is important for actor to remember the observation space
            do not remove local done

            AS OF 17 / 05 / 2021
            IMPORTANT!!!!!11
            DO NOT REMOVE LOCAL DONE 
            need to figure out how to set local done cleanly

            do not try to use lambda function to access it looks like it could work but it does not
            read more about lambda PYTHON function
        '''
        rledcss.setSimulationDone(False)
        if rledcss.simulationStart:
            for i in range(max_steps):
                count += 1
                
                # The state imagine the 'future'
                imagination = actor.imagine(state)
                #imagine_action = np.argmax(imagination[0])
                
                #action = actor.act(state, imagine_state)
                #action = actor.act(state)
                
                prev_state = state
                next_state, reward, done, _ = imaginationEnv.step(imagination)
                #next_state, reward, _ = imaginationEnv.step(imagine_action)
                score += reward

                next_state = convert_target_to_real_state(next_state, imaginationEnv.state_space)
                #imaginationEnv.done = rledcss.done
                actor.remember(state, imagination, reward, next_state, done)
                #rollout.insert(i, state, imagine_action, reward)
                state = next_state

                #print(f'Score {score}')

                # Set Graph attribute (graph1)
                #time2 = imaginationEnv.getTime()
                #time2 += 1

                
                ln.set_color('red')
                #x = np.append(x , float(count))
                #x = np.append(x , float(ImaginationEnvironment.getTime()))
                x = np.append(x , float(imaginationEnv.getTime()))
                y = np.append(y , float(reward))
                ln.set_xdata(x)
                ln.set_ydata(y)
                ax[0].plot(x, y)
                

                # Windowed graph view
            
                if imaginationEnv.getTime() < imagineDepth:
                    ax[0].set_xlim(0, imagineDepth + imaginationEnv.getTime())

                else:
                    ax[0].set_xlim(imaginationEnv.getTime() - imagineDepth, imagineDepth + imaginationEnv.getTime() - 45)
                    plt.draw()
                
                fr_number.set_text("Simulation Cycle : {e}".format(e=e+1))
                
                # update graph maker
                gm.update()
                
                if rledcss.activated:
                    rledcss.setAgentActivated(False)
                    #action = actor.act(next_state, imagination) 
                    action = actor.imagine(state)             
                    #rledcss.setAgentAction(action)
                    #plt.plot(count,0, 'ro')
                    #c = convert_action_toReadable(action)
                    #m *= -1
                    ann = ax[0].annotate("Agent Called", (imaginationEnv.getTime(), reward), arrowprops=dict(facecolor='black', shrink=0.05))
                    ann_list.append(ann)
                    plt.draw()
                    #print("Action ",action)
                if params['batch_size'] > 1:
                    actor.replay()
                if rledcss.done:
                    rledcss.setSimulationReady(True)
                    #plt.removeAll()
                    for i, a in enumerate(ann_list):
                        a.remove()
                    ann_list[:] = []

                    x2 = np.append(x2 , float(e+1))
                    y2 = np.append(y2 , float(score))
                    ax[1].plot(x2, y2)
                    plt.draw()
                    #gm.setPreviousLineToDiff()



                    
                    '''
                        HOW THE LINES SET AGAIN
                        REMINDER to note graph execution

                        AS OF 11 / 05 / 2021
                        MULTI COLOUR LINE is no longer functioning
                    '''
                    #print("Time : ",imaginationEnv.getTime())
                    #plt.cla()
                    #gm.removeAllArt()
                    #x = np.array([0], dtype = np.float32)
                    #y = np.array([0], dtype = np.float32)
                    #gm.removeOldSetNew(ax.plot(x, y, color='black', animated=True))
                    #plt.cla()
                    #gm.add_artist([ax.plot(0, 0, color='black', animated=True), fr_number])
                    #ln._axes.lines.clear()
                    #if ln is not None:
                    #   ln.set_color('gray')
                    #gm.update()
                    sum_sla_violation += rledcss.sla_violation
                    sum_agent_call += rledcss.agent_call
                    # Only allow the system to edit the log field
                    '''sidebar information'''
                    text_area.configure(state="normal")
                    text_area.insert('insert',f'{e+1}/{imagineDepth}: {round(score,2)} : {rledcss.agent_call} : {rledcss.sla_violation}' + '\n')
                    #text_area.insert('insert',f'{e+1}/{imagineDepth}: {round(score,2)} : {rledcss.agent_call} : {rledcss.sla_violation}' + '\n')
                    text_area.configure(state="disabled")
                    text_area.see("end")
                    #actor.remember(state, imagine_action, reward, next_state, True)
                    actor.remember(state, imagination, reward, next_state, True)
                    #print(f'Final Cloud State: {str(prev_state)}')
                    #print(f'Imagination Depth: {e+1}/{imagineDepth}, Score: {score}')
                    if (e + 1) == ((imagineDepth / 10) * num_g):
                        plt.savefig(f'Graph {e+1}_{imagineDepth}')
                        num_g += 1
                    break
            #end mark: for i in range(max_steps)
        #returns = rollout.compute_returns(next_state, params["gamma"])
        sum_of_rewards.append(score)
        #print(f'Returns : {returns}')
    rledcss.setSimulationStart(False)
    maxRank = max(y2)
    totalReward = sum(sum_of_rewards)
    lrt = maxRank - totalReward/imagineDepth

    text_area_lrt.config(state="normal")
    text_area_lrt.insert('insert',f'LRT : {round(lrt, 2)}'+ '\n')
    text_area_lrt.insert('insert',f'Max Reward: {round(maxRank, 2)}'+ '\n')
    text_area_lrt.insert('insert',f'Total Reward : {round(totalReward, 2)}'+ '\n')
    text_area_lrt.insert('insert',f'Total Agent Call : {sum_agent_call}'+ '\n')
    text_area_lrt.insert('insert',f'Total SLA Violation : {sum_sla_violation}'+ '\n')
    #text_area_lrt.insert('insert',f'Total Simulation Time : {imaginationEnv.getTime()}'+ '\n')
    text_area_lrt.configure(state="disabled")
    print("Sum of Reward : ", sum(sum_of_rewards))
    print("Total Agent Call :", sum_agent_call) #A-add
    print("Total SLA violation :", sum_sla_violation) #A-add
    print("Simulated Time :", imaginationEnv.getTime()) #A-add
   

    plt.savefig('Complete Graph')
    return actor

def convert_action_toReadable(action):
    if action == 0:
        return "Called:Scale Down";
    if action == 1:
        return "Called:Scale Up";
    if action == 2:
        return "Called:No Action";

def convert_target_to_real_state(next_state, state_space):
    _state = next_state
    _state = np.reshape(_state, (1, state_space))

    return _state

'''
    This function is used to delay python until CloudSim Plus has created vms datacenter host broker etc
'''
def delayUntilReady():
    if rledcss.ready:
        return True
    for e in range(10000000):
        if rledcss.ready:
            return True
    return False

def buttonFunction(actor, imagineDepth, imaginationEnv, gm, gateway, host, vm, cloudlet, maxPower, idlePower, time, ax):
    '''
        Set the param first/before starting simulation to avoid catching the default sets 
    '''
    setParam = host.get() + "," + vm.get() + "," + cloudlet.get() + "," + maxPower.get() + "," + idlePower.get() + "," + time.get()
    rledcss.setSimulationSettings(setParam)
    print (setParam)
    
    rledcss.setSimulationStart(True)
    #print (cloudlet.get()) 
    #print (maxPower.get()) 
    #print (idlePower.get()) 
    #print (time.get()) 

    '''
    This is to delay Python while waiting for CloudSim Plus to create VMs, Host, Datacenter, broker and etc.
    '''
        
    # critic_train(actor, int(imagineDepth), imaginationEnv, gm, gateway, ax)
    '''
    This is to delay Python while waiting for CloudSim Plus to create VMs, Host, Datacenter, broker and etc.
    '''

    loop = asyncio.get_event_loop()
    all_threads = []
    #berapa number worker nak / async(launch sekali semua)
    for i in range(2):
        thr = threading.Thread(target=wrapper,args=(loop, actor, int(imagineDepth), imaginationEnv, gm,gateway,ax))
        thr.start()
        all_threads.append(thr)
    # [i.join() for i in all_threads] 
    print("Done")
    rledcss.setSimulationStart(False)
    #print ("Button")
    
'''
As of 16 / 05 / 2021
the function createEntry() is no longer used
remember to remove latter

def createEntry(window, textV, labelT, defaultT):
    tempL = Label(window, text=labelT)
    tempL.pack(side="left")
    tempT = ttk.Entry(window, width=15, textVariable=textV)
    tempT.insert("insert",defaultT)
    tempT.pack(side="left")
'''

# RLEDCSSInterface are declared outside of any block to allow global access
rledcss = RLEDCSS()

ln = None
lines = None
if __name__ == '__main__':

    params = dict()
    try:
        with open('setting.txt') as f:
            lines = f.readlines()
    except:
        messagebox.showerror("Error", "setting.txt is not found")

    #Parameter for Keras-tensorflow
    params['name'] = lines[0].split(':')[1]
    params['epsilon'] = int(lines[1].split(':')[1])
    params['gamma'] = float(lines[2].split(':')[1])
    params['batch_size'] = int(lines[3].split(':')[1])
    params['epsilon_min'] = float(lines[4].split(':')[1])
    params['epsilon_decay'] = float(lines[5].split(':')[1])
    params['learning_rate'] = float(lines[6].split(':')[1])
    params['layer_sizes'] = [128, 128, 128]

    results = dict()
    
    imagineDepth = 50

     # Set imagination observation info
    imaginationInfos = {'States: Minimum Threshold': {'state_space':''},
                        'States: Inbetween Threshold':{'state_space':''},
                        'States: Maximum Threshold':{'state_space':''},
                        'States: Inbetween Threshold': {'state_space':''},
                        'States: Greater VM usage':{'state_space':''},
                        'States: Greater Host usage':{'state_space':''},
                        'States: MIPS left':{'state_space':''},
                        'States: Zero Power usage':{'state_space':''}}
    
    imagineEnv = ImaginationEnvironment(1)

    # Established connection with JAVA first
    '''
        PLEASE REMEMBER THAT THE GATEWAY USED TO GET THE SIMULATION MUST BE THE SAME GATEWAY THAT INITIATE PYTHON
        and yes PYTHON need to run first ..... refers FYP Operation Draft.txt

        AS OF 12 / 05 / 2021
        JAVA will need to run first because callback server must be TRUE in order to accept connection
        the same requiremnt is still needed for the gateway 

        AS OF 17 / 05 / 2021
        OK FOR THE LAST TIME PYTHON will need to run first, callback server can be trigger by PYTHON for JAVA
        PYTHON and JAVA connection is initiated by the same Gateway without extruding the Java Virtual Machine (JVM)

        NOTE : JVM need to gain full package in order to parse the full java object to the collection

        NOTE : When returning to an interface the self. must be correctly formatted

        NOTE : If Java error _get_id() -> check return format in Interface, sometimes the default for PYTHON might not be equal to the JAVA @param

        NOTE : FYP Operation Draft.txt is still referred
        
    '''
    gateway = JavaGateway(
        callback_server_parameters=CallbackServerParameters(),
        auto_convert=True,
        python_server_entry_point=rledcss)

    actor = ActorCritic(imagineEnv, params)

    # Build the Window
    window = tkinter.Tk();
    winStyle = ttk.Style();
    winStyle.theme_use('clam');
    window.title("Cloud Auto Scaler");
    window.geometry("1080x640");
    window.configure(background = '#03254C');
    window.resizable(0,0);
    window.focus_set();

    frame = tkinter.Frame(window)
    frame.pack(fill="x", expand=True)

    # Set initial for both graph

    x = np.array([0], dtype = np.float32)
    y = np.array([0], dtype = np.float32)

    # Set an extend graph
    '''
    plt2 = plt
    fig2, ax2 = plt2.subplots()
    canvas2 = FigureCanvasTkAgg(fig2, master = window)
    canvas2.get_tk_widget().place(x=650,y=0)
    # Set the extend 
    plt2.ylabel("LRT")
    plt2.xlabel("Simulation Cycle")

    (ln2,) = ax2.plot(x,y, color="blue", animated=True)
    '''
    # Set up canvas to draw the graph
    fig, ax = plt.subplots(2,1)
    canvas = FigureCanvasTkAgg(fig, master = window)
    #canvas.get_tk_widget().pack(side="left")
    canvas.get_tk_widget().place(x=0,y=0)
    #canvas.get_tk_widget().config(width=640, height=580)

    # plot returns a list with line instances, one for each line you draw,
    (ln,) = ax[0].plot(x, y, color='black', animated=True)
    
    # Set the current graph 
    ax[0].set_ylim(-3, 3)
    ax[0].set_xlim(0,imagineDepth)
    #plt.suptitle("Current")
    ax[0].set_ylabel("Reward")
    ax[0].set_xlabel("Simulation Time")

    # Add a frame number
    fr_number = ax[0].annotate("0",
                            (0, 1),
                            xycoords="axes fraction",
                            xytext=(10, -10),
                            textcoords="offset points",
                            ha="left",
                            va="top",
                            animated=True,
    )

    #plot the extended graph
    ax[1].plot(x,y, color="red", animated=True)

    ax[1].set_ylabel("Total Reward")
    ax[1].set_xlabel("Simulation Cycle")

    # Init artist to draw
    canvas.draw();
    gm = GraphMaker.GraphMaker(fig.canvas, [ln, fr_number])
    plt.ion()
    plt.tight_layout()
    #plt2.ion()
    #canvas.pause(0.05)
    #plt.pause(0.05)
    #plt.show(block = False)

    #Create ScrolledText
    text_area = scrolledtext.ScrolledText(window, wrap = tkinter.WORD, width = 41, height = 23, font = ("Courier New",15))
    text_area.configure(state="disabled")
    #text_area.pack(side="right")
    text_area.place(x=640, y=0)

    text_area_lrt = scrolledtext.ScrolledText(window, wrap = tkinter.WORD, width = 41, height = 8, font = ("Courier New",14))
    text_area_lrt.insert('insert','SIMULATION SUMMARY\n')
    text_area_lrt.configure(state="disabled")
    #text_area.pack(side="right")
    text_area_lrt.place(x=640, y=485)

    # Create the String variables first
    host = tkinter.StringVar()
    vm = tkinter.StringVar()
    cloudlet = tkinter.StringVar()
    maxPower = tkinter.StringVar()
    idlePower = tkinter.StringVar()
    time = tkinter.StringVar()

    # Create the button for 
    #plot_button = Button(master = window, command = testButtonFunction, height = 3, width = 20,text = "Start Simulation")
    #lambda: action(someNumber)
    start_button = Button(master = window, command = lambda: buttonFunction(actor,episode.get(), imagineEnv, gm, gateway,host, vm, cloudlet, maxPower, idlePower, time, ax), height = 3, width = 20,text = "Start Simulation")
    start_button.place(x=0, y=570)

    guiL = Label(window, text="Simulation Parameter Setting")
    #guiL.pack(side="top")
    guiL.place(x=230, y=510)
    

    # Add Reinforcement Learning Settings
    episode = tkinter.StringVar()
    episodeL = Label(window, width = 6, text="Cycle")
    #episodeL.pack(side="left")
    episodeL.place(x=3, y=540)
    episodeT = ttk.Entry(window, width = 5, textvariable = episode)
    episodeT.insert("insert", "50")
    #episodeT.pack(side="left")
    episodeT.place(x=51, y=540)

    hostL = Label(window, width = 6, text="Host")
    #hostL.pack(side="left")
    hostL.place(x=80, y = 540)
    hostT = ttk.Entry(window, width = 5, textvariable = host)
    hostT.insert("insert", "1")
    #hostT.pack(side="left")
    hostT.place(x=125, y=540)

    vmL = Label(window, width = 6, text="VM(s)")
    #vmL.pack(side="left")
    vmL.place(x=160, y=540)
    vmT = ttk.Entry(window, width = 5, textvariable = vm)
    vmT.insert("insert", "2")
    #vmT.pack(side="left")
    vmT.place(x=205, y=540)

    cloudletL = Label(window, width = 8, text="Cloudlet")
    #cloudletL.pack(side="left")
    cloudletL.place(x=240, y=540)
    cloudletT = ttk.Entry(window, width = 5, textvariable = cloudlet)
    cloudletT.insert("insert", "4")
    #cloudletT.pack(side="left")
    cloudletT.place(x=300, y=540)

    maxPowerL = Label(window, width = 9,text="Max Power")
    #maxPowerL.pack(side="left")
    maxPowerL.place(x=335, y=540)
    maxPowerT = ttk.Entry(window, width = 5, textvariable = maxPower)
    maxPowerT.insert("insert", "50")
    #maxPowerT.pack(side="left")
    maxPowerT.place(x=400, y=540)
    
    idlePowerL = Label(window, width = 9, text="Idle Power")
    #idlePowerL.pack(side="left")
    idlePowerL.place(x=435, y=540)
    idlePowerT = ttk.Entry(window, width = 5, textvariable = idlePower)
    idlePowerT.insert("insert", "30")
    #idlePowerT.pack(side="left")
    idlePowerT.place(x=500, y=540)

    timeL = Label(window, width = 10, text="Sim Time")
    #timeL.pack(side="left")
    timeL.place(x=530, y=540)
    timeT = ttk.Entry(window, width = 6, textvariable = time)
    timeT.insert("insert", "3600")
    #timeT.pack(side="left")
    timeT.place(x=600,y=540)

    window.mainloop()
    
    #print(cloudlet.get())
    
    #sum_of_rewards = critic_train(imagineDepth,  imagineEnv, gm)
    #results[params['name']] = sum_of_rewards

    #matplot lib kne buat semula
    #plot saloh
    #plot_result(results, direct=True, k=20)
