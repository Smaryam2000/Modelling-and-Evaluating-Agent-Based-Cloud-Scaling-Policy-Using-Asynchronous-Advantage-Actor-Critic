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
from environment import ExperienceEnvironment
from Functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from py4j.java_gateway import (JavaGateway, CallbackServerParameters)
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from plot_script import plot_result
tf.config.threading.set_inter_op_parallelism_threads(
    2
)
from tqdm import tqdm
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
    masterWeights = np.array(masterActor.model.get_weights())
    workerWeights = np.array(actor.model.get_weights())
    if np.sum(masterWeights[0]) > 0:
        p = 0.7
        masterWeights *= p
        workerWeights *= (1-p)
        weights = (masterWeights+workerWeights)
    else:
        weights = workerWeights
    masterActor.model.set_weights(weights.tolist())
    print("*****************update weights*****************")

'''looping for worker training'''
def wrapper(loop,*args):
    callback(loop.run_until_complete(critic_train(*args)))


class RLEDCSS(object):
    start = False;
    ready = False;  #Give Java time to set up
    done = False;
    activated = False;
    action = 0
    sla_violation = 0
    agent_call = 0
   # setParam = "1,2,4,50,30,3600"
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


async def critic_train(actor, trainDepth, experienceEnv, gm, gateway, ax):
    # Actor will be initialized first before training
    # actor = ActorCritic(experienceEnv, params)   
    global masterActor
    
    if masterActor is None:
        masterActor = ActorCritic(experienceEnv, params)
    actor = ActorCritic(experienceEnv, params)

    actor.model = tf.keras.models.clone_model(
        masterActor.model, input_tensors=None, clone_function=None
    )                   
                      
    
    # Initialize CSP connection Enviroment
    trainingEnv.getSimulation(gateway);

    # This variable is used to mark the graph
    num_g = 1
    
    sum_of_rewards = []
    sum_of_losses = []
    sum_of_powers = []

    max_steps = 10000

    #contain the total number of SLA violation for all episode
    sum_sla_violation = 0
    sum_agent_call = 0


    ann_list = []
    #x = np.array([0], dtype = np.float32)
    #y = np.array([0], dtype = np.float32)

    x2 = np.array([0], dtype = np.float32)
    y2 = np.array([0], dtype = np.float32)

    
    #x3 = np.array([0], dtype = np.float32)
    #y3 = np.array([0], dtype = np.float32)
    
    #rollout = RolloutStorage(max_steps, experienceEnv.state_space.shape)

    #Line 11
    for e in range(trainDepth):
        print("#"*24)
        print(f"depth: {e}")
        print("#"*24)
        state = experienceEnv.reset()
        state = convert_target_to_real_state(state, experienceEnv.state_space)
        score = 0
        count = 0
        energy= 0
        max_steps = 200

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
        start = t.monotonic()
        if rledcss.simulationStart:
            tbar = tqdm(range(max_steps))
            for i in tbar:
                count += 1
                
                experience= actor.train(state)
                #train_action = np.argmax(experience[0])
                
                #action = actor.act(state, train_state)
                #action = actor.act(state)
                
                prev_state = state
                next_state, reward, done, _ = experienceEnv.step(experience)
                #next_state, power, reward, _ = experienceEnv.step(train_action)
                score += reward
                #energy += power
               
                tbar.set_postfix_str(s=str(experience))
                
                next_state = convert_target_to_real_state(next_state, experienceEnv.state_space)
                #experienceEnv.done = rledcss.done
                #actor.remember(state, experience, reward, next_state, done)
                #rollout.insert(i, state, train_action, reward)
                state = next_state

                #print(f'Score {score}')

                # Set Graph attribute (graph1)
                #time2 = experienceEnv.getTime()
                #time2 += 1

                
                ln.set_color('red')
                #x = np.append(x , float(count))
                #x = np.append(x , float(ExperienceEnvironment.getTime()))
                # x = np.append(x , float(experienceEnv.getTime()))
                x = np.append(x , float(t.monotonic()-start))
                y = np.append(y , float(reward))
                ln.set_xdata(x)
                ln.set_ydata(y)
                ax[0].plot(x, y)
                

                # Windowed graph view
            
                ax[0].set_xlim(0, trainDepth + x[-1])
                # if experienceEnv.getTime() < trainDepth:
                #     ax[0].set_xlim(0, trainDepth + experienceEnv.getTime())

                # else:
                #     ax[0].set_xlim(experienceEnv.getTime() - trainDepth, trainDepth + experienceEnv.getTime() - 45)
                plt.draw()
                
                fr_number.set_text("Episode : {e}".format(e=e+1))
                
                # update graph maker
               # if i%2000 == 0:
                gm.update()
                
                if rledcss.activated:
                    rledcss.setAgentActivated(False)
                    #action = actor.act(next_state, imagination) 
                    action = actor.train(state)             
                    rledcss.setAgentAction(action)
                    #plt.plot(count,0, 'ro')
                    #c = convert_action_toReadable(action)
                    #m *= -1
                    ann = ax[0].annotate("Agent Called", (x[-1], reward), arrowprops=dict(facecolor='black', shrink=0.05))
                    ann_list.append(ann)
                    plt.draw()
                    #print("Action ",action)
                # if params['batch_size'] > 1:
                #     t.sleep(1)
                #     actor.experience()
                if rledcss.done:
                    rledcss.setSimulationReady(True)
                    #plt.removeAll()
                    for i, a in enumerate(ann_list):
                        a.remove()
                    ann_list[:] = []

                    

               
                    #gm.setPreviousLineToDiff()



                    
                    '''
                        HOW THE LINES SET AGAIN
                        REMINDER to note graph execution

                        AS OF 11 / 05 / 2021
                        MULTI COLOUR LINE is no longer functioning
                    '''
                    #print("Time : ",experienceEnv.getTime())
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
                    text_area.insert('insert',f'{e+1}/{trainDepth}: {round(score,2)} : {rledcss.agent_call} : {rledcss.sla_violation}' + '\n')
                    #text_area.insert('insert',f'{e+1}/{trainDepth}: {round(score,2)} : {rledcss.agent_call} : {rledcss.sla_violation}' + '\n')
                    text_area.configure(state="disabled")
                    text_area.see("end")
                    #actor.remember(state, train_action, reward, next_state, True)
                    #actor.remember(state, experience, reward, next_state, True)
                    #print(f'Final Cloud State: {str(prev_state)}')
                    #print(f'Imagination Depth: {e+1}/{trainDepth}, Score: {score}')
                    if (e + 1) == ((trainDepth / 10) * num_g):
                        plt.savefig(f'Graph {e+1}_{trainDepth}')
                        num_g += 1
                    break
            #end mark: for i in range(max_steps)
            x2 = np.append(x2 , float(e+1))
            y2 = np.append(y2 , float(score))
            ax[1].plot(x2, y2)
            plt.draw()

            #x3 = np.append(x3 , float(e+1))
            #y3 = np.append(y3 , float(power))
            #ax[1].plot(x3, y3)
            #plt.draw()
        #returns = rollout.compute_returns(next_state, params["gamma"])
        sum_of_rewards.append(score)
        #print(f'Returns : {returns}')
    rledcss.setSimulationStart(False)
    maxRank = max(y2)
    totalReward = sum(sum_of_rewards)
    lrt = maxRank - totalReward/trainDepth

    text_area_lrt.config(state="normal")
    text_area_lrt.insert('insert',f'LRT : {round(lrt, 2)}'+ '\n')
    text_area_lrt.insert('insert',f'Max Reward: {round(maxRank, 2)}'+ '\n')
    text_area_lrt.insert('insert',f'Total Reward : {round(totalReward, 2)}'+ '\n')
    text_area_lrt.insert('insert',f'Total Agent Call : {sum_agent_call}'+ '\n')
    text_area_lrt.insert('insert',f'Total SLA Violation : {sum_sla_violation}'+ '\n')
    #text_area_lrt.insert('insert',f'Total Simulation Time : {experienceEnv.getTime()}'+ '\n')
    text_area_lrt.configure(state="disabled")
    print("Sum of Reward : ", sum(sum_of_rewards))
    print("Total Agent Call :", sum_agent_call) #A-add
    print("Total SLA violation :", sum_sla_violation) #A-add
    print("Simulated Time :", experienceEnv.getTime()) #A-add

    plt.savefig('Complete Graph')
    return actor

def convert_action_toReadable(action):
    if action == 0:
        return "Called:Scale Out";
    if action == 1:
        return "Called:Scale In";
    if action == 2:
        return "Called:No Action";
    #if action == 3:
     #       return "Called:Scale In";
    #if action == 4:
     #       return "Called:No Action";

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

def buttonFunction(actor, trainDepth, experienceEnv, gm, gateway, host, vm, cloudlet, maxPower, idlePower, time, ax):
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
        
    # critic_train(actor, int(trainDepth),experienceEnv , gm, gateway, ax)
   

    loop = asyncio.get_event_loop()
    all_threads = []
    #berapa number thread nak / async(launch sekali semua)
    for i in range(1):
        thr = threading.Thread(target=wrapper,args=(loop, actor, int(trainDepth), experienceEnv, gm,gateway,ax))
        thr.start()
        all_threads.append(thr)
    # [i.join() for i in all_threads] 
    # thr = threading.Thread(target=wrapper,args=(loop, actor, int(trainDepth), experienceEnv, gm,gateway,ax))
    # thr.start()
    print("Done")
    rledcss.setSimulationStart(False)
    print ("Button")
    
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
    
    trainDepth = 50

     # Set imagination observation info
    imaginationInfos = {'States: Minimum Threshold': {'state_space':''},
                        'States: Inbetween Threshold':{'state_space':''},
                        'States: Maximum Threshold':{'state_space':''},
                        'States: Inbetween Threshold': {'state_space':''},
                        'States: Greater VM usage':{'state_space':''},
                        'States: Greater Host usage':{'state_space':''},
                        'States: MIPS left':{'state_space':''},
                        'States: Zero Power usage':{'state_space':''}}
    
    trainingEnv = ExperienceEnvironment(1)

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

    actor = ActorCritic(trainingEnv, params)
    #GUI
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
    ax[0].set_xlim(0,trainDepth)
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
    ax[1].set_xlabel("Simulation Episode")

    #ax[2].plot(x,y, color="blue", animated=True)

    #ax[2].set_ylabel("Power")
    #ax[2].set_xlabel("Simulation Episode")


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
    start_button = Button(master = window, command = lambda: buttonFunction(actor,episode.get(), trainingEnv, gm, gateway,host, vm, cloudlet, maxPower, idlePower, time, ax), height = 3, width = 20,text = "Start Simulation")
    start_button.place(x=0, y=570)

    guiL = Label(window, text="Simulation Parameter Setting")
    #guiL.pack(side="top")
    guiL.place(x=230, y=510)
    

    # Add Reinforcement Learning Settings
    episode = tkinter.StringVar()
    episodeL = Label(window, width = 6, text="Episode")
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
    
    