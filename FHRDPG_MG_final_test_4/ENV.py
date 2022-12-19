# environment for microgrid


import numpy as np
import time
import matplotlib.pyplot as plt
import random

class env_mg:
    def __init__(self,):
        self.dg_max=600 # DG max power，kW
        self.dg_min=100 # DG min power，kW
        self.action_max = np.array(self.dg_max)
        self.action_min = np.array(self.dg_min)
        self.dt=1 # time step 1/4h=15min
        self.PE_max=120 # battery max power，kW
        self.E_max=2000 # battery max capacity，kWh
        self.E_min=24 # battery min capacity，kWh
        self.eta_ch=0.98 # charging efficiency
        self.eta_dis=0.98 # discharging efficiency
        self.PV_max=300 # PV max power
        self.PV_min=0 # PV min power
        self.load_max= 1500 # load max power
        self.load_min= 10 # load min power
        self.actionDim=1 # action space dimensionality
        self.stateDim=2 # state space dimensionality
        self.hisStep=4 # history time steps，RDPG
        self.cus=[]
        self.cdg=[]
        self.state=[]
        self.obs=[]
        self.PE_ch_lim=self.PE_max
        self.PE_dis_lim = self.PE_max
        self.rewardScale = 2e-3
        self.seed = 6
        self.ad = 0.005
        self.bd = 6
        self.cd = 100
        self.k1 = 1
        self.k2 = 1
        self.k3 = 1/1000
        # self.PL = []
        # self.PPV = []

        np.random.seed(self.seed)
        random.seed(self.seed)

    def step(self,u):
        dt = self.dt
        # PL_t = self.state[0]
        # PPV_t = self.state[1]
        DPVL_t = self.state[0]
        # 开始计算costs
        ad = self.ad
        bd = self.bd
        cd = self.cd
        k3 = self.k3
        CDG_t = k3 * (ad * u ** 2 + bd * u + cd) * dt
        self.cdg.append(CDG_t/k3)

        # E_t = self.state[2] # E_t_1是E_t-1，E_t1是E_t+1
        E_t = self.state[1]

        PE_ch_lim = min(self.PE_max, (self.E_max-E_t)/self.eta_ch/dt)
        PE_dis_lim = min(self.PE_max, self.eta_dis*(E_t-self.E_min)/dt)
        self.PE_ch_lim = PE_ch_lim
        self.PE_dis_lim = PE_dis_lim
        # delta_t = u + PPV_t - PL_t
        delta_t = u - DPVL_t 
        # print ("delta_t:", delta_t)
        # print ("u:", u)
        k1 = self.k1
        k2 = self.k2
        if delta_t > PE_ch_lim:
            CUS_t = k1 * (delta_t - PE_ch_lim) * dt
            self.cus.append(CUS_t / k1)
        elif delta_t < -PE_dis_lim:
            CUS_t = -k2 * (delta_t + PE_dis_lim) * dt
            self.cus.append(-CUS_t / k2)
        else:
            CUS_t = 0
            self.cus.append(0)
        
        costs = CDG_t + CUS_t
        # costs = CUS_t
        # costs error
        if costs<0:
            raise Exception("costs equals to ",costs, ", costs should not be less than 0!")

        if delta_t >= 0:
            u_t = 1
            PE_t = min(delta_t,PE_ch_lim)
        else:
            u_t = 0
            PE_t = min(-delta_t,PE_dis_lim)
        E_t1 = E_t + self.eta_ch * u_t * PE_t * dt - (1-u_t) * PE_t * dt / self.eta_dis

        if E_t1 < self.E_min:
            E_t1 = self.E_min
        elif E_t1 > self.E_max:
            E_t1 = self.E_max
        
        # read data
        # f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        # lines = f.readlines()
        # cnt = self.cnt
        # PL_t = float(lines[cnt].strip().split(',')[-2])
        # PPV_t = float(lines[cnt].strip().split(',')[-1])
        self.cnt += 1
        PL_t = float(self.PL[self.cnt])
        PPV_t = float(self.PPV[self.cnt])
        PL_t_1 = float(self.PL[self.cnt-1])
        PPV_t_1 = float(self.PPV[self.cnt-1])
        # derive state，PL_t , PPV_t from data
        # state = np.array([PL_t, PPV_t , E_t1])
        state = np.array([PL_t - PPV_t , E_t1])
        self.state = state
        # derive observations，PL_t_1 , PPV_t_1 from history data
        obs = np.array([PL_t_1 - PPV_t_1 , E_t1])
        self.obs = obs
    #    return state, -costs, False, {}
        return obs, -costs, False, {}


    def load_data(self, n_days):
        f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        lines = f.readlines()
        # n = 2071 + 96*7
        # n = 2071
        # n = 517
        n = 517 + 14*24
        t_max = 24 * n_days
        self.PL = np.zeros(shape=[t_max+1+self.hisStep])
        self.PPV = np.zeros(shape=[t_max+1+self.hisStep])
        for i in range(-self.hisStep, t_max+1):
            PL_t = float(lines[n+i].strip().split(',')[-2])
            PPV_t = float(lines[n+i].strip().split(',')[-1])
            self.PL[i+self.hisStep] = PL_t
            self.PPV[i+self.hisStep] = PPV_t
        PL_PPV = self.PL - self.PPV
        PL_PPV_max = np.max(PL_PPV)
        PL_PPV_min = np.min(PL_PPV)
        self.PL_PPV_bias = (PL_PPV_max + PL_PPV_min) / 2.0
        self.PL_PPV_range = PL_PPV_max - self.PL_PPV_bias
        self.E_bias = (self.E_max + self.E_min) / 2.0
        self.E_range = self.E_max - self.E_bias

    def get_range_bias(self, n_days):
        f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        lines = f.readlines()
        # n = 2071 + 96*7
        # n = 2071
        n = 517 - 24*14
        # n = 517
        t_max = 24 * n_days
        PL = np.zeros(shape=[t_max])
        PPV = np.zeros(shape=[t_max])
        for i in range(t_max):
            PL_t = float(lines[n+i].strip().split(',')[-2])
            PPV_t = float(lines[n+i].strip().split(',')[-1])
            PL[i] = PL_t
            PPV[i] = PPV_t
        PL_PPV = PL - PPV
        PL_PPV_max = np.max(PL_PPV)
        PL_PPV_min = np.min(PL_PPV)
        self.PL_PPV_bias = (PL_PPV_max + PL_PPV_min) / 2.0
        self.PL_PPV_range = PL_PPV_max - self.PL_PPV_bias
        self.E_bias = (self.E_max + self.E_min) / 2.0
        self.E_range = self.E_max - self.E_bias

    def reset(self,t):
        # f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        # lines=f.readlines()
        # np.random.seed(int(time.time()))
        # np.random.seed(123)
        # n=np.random.random_integers(1,len(lines)-24/self.dt)
        # n=24900
        # n=128
        # n = 2071 + 96*7 + t
        # n = 2071 + 96 * 7 + 58 + t
        # n = 2071 + t
        # print('Start from line ' + str(n) + ' in file data_dt_' + str(int(60 * self.dt)) + '.csv')
        # self.cnt = n
        # PL_t = float(lines[n].strip().split(',')[-2])
        # PPV_t = float(lines[n].strip().split(',')[-1])
        PL_t = float(self.PL[t+self.hisStep])
        PPV_t = float(self.PPV[t+self.hisStep])
        PL_t_1 = float(self.PL[t-1+self.hisStep])
        PPV_t_1 = float(self.PPV[t-1+self.hisStep])
        # self.cnt += 1
        self.cnt = t+self.hisStep
        # self.E_t = np.random.random_integers(self.E_min, self.E_max)
        E_t = random.randint(self.E_min, self.E_max)
        # self.E_t = self.E_min
        # self.E_t = 20
        # state = np.array([PL_t, PPV_t, E_t])
        state = np.array([PL_t - PPV_t, E_t])
        self.state = state
        # return state
        obs = np.array([PL_t_1 - PPV_t_1, E_t])
        self.obs = obs
        return obs

    def myopic_policy(self, state):
        ad = self.ad
        bd = self.bd
        cd = self.cd
        # PL_t = state[0]
        # PV_t = state[1]
        DPVL_t = state[0]
        # E_t = state[2]
        E_t = state[1]
        k3 = self.k3

        PE_dis_lim = min(self.PE_max, self.eta_dis * (E_t - self.E_min) / self.dt)
        # if PL_t - PV_t - PE_dis_lim > self.dg_max:
        if DPVL_t - PE_dis_lim > self.dg_max:
            a_T = self.dg_max
            # CUS_T = PL_t - PV_t - PE_dis_lim - self.dg_max
            CUS_T = (DPVL_t - PE_dis_lim - self.dg_max) * self.dt
        # elif PL_t - PV_t - PE_dis_lim <= self.dg_max and PL_t - PV_t - PE_dis_lim >= self.dg_min:
        elif DPVL_t - PE_dis_lim <= self.dg_max and DPVL_t - PE_dis_lim >= self.dg_min:
            # a_T = PL_t - PV_t - PE_dis_lim
            a_T = DPVL_t - PE_dis_lim
            CUS_T = 0
        else:
            a_T = self.dg_min
            # if PL_t >= PV_t + self.dg_min:
            if DPVL_t >= self.dg_min:
                CUS_T = 0
            else:
                # CUS_T = PV_t + self.dg_min - PL_t
                CUS_T = (self.dg_min - DPVL_t) * self.dt
        CDG_t = k3 * (ad * a_T ** 2 + bd * a_T + cd) * self.dt
        self.cus.append(CUS_T)
        self.cdg.append(CDG_t)

        # new_state_T, r_T, done_T, _ = env.step(a_T)
        costs = CDG_t + CUS_T
        r_T = float(-costs)
        r_T *= self.rewardScale

        return r_T, a_T

    def getHistory(self, t):
        history = np.zeros(self.hisStep * self.stateDim)
        state_bias = np.array([self.PL_PPV_bias, self.E_bias])
        state_range = np.array([self.PL_PPV_range, self.E_range])
        for i in range(self.hisStep):
            his_state = [self.PL[t - 1 + i] - self.PPV[t - 1 + i], 0]
            his_state_norm = (his_state - state_bias) / state_range
            history[i * self.stateDim: (i + 1) * self.stateDim] = np.array(his_state_norm)
        self.cnt = self.hisStep
        return history


    def baseline_random_run(self, num_episode):
        f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        lines = f.readlines()
        n = np.random.random_integers(1,len(lines) - 24 / self.dt)
        # print('Start from line'+str(n)+' in file data_dt_' + str(int(60 * self.dt)) + '.csv')
        self.cnt = n
        pv = []
        pl = []
        e = []

        dg = []
        rt = []
        cus = []
        cdg = []
        for i in range(num_episode):
            o = self.reset()
            totalReward = 0
            for j in range(int(24/self.dt)):
                np.random.seed(int(time.time()+j))
                a = np.random.random_integers(self.dg_min, self.dg_max)
                oPrime, r, d, _ = self.step(a)
                r *= self.rewardScale
                totalReward += r
                pl.append(oPrime[0])
                pv.append(oPrime[1])
                e.append(oPrime[2])
                dg.append(a)
                cus.append(self.cus[j])
                cdg.append(self.cdg[j])
            print("Episode:",i,",score:",totalReward)
            x = range(0, int(24/self.dt))
            plt.subplot(611)
            plt.plot(x, pv, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("PV")

            plt.subplot(612)
            plt.plot(x, pl, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("Load")

            plt.subplot(613)
            plt.plot(x, e, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("battery SOC")

            plt.subplot(614)
            plt.plot(x, dg, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("DG power")

            plt.subplot(615)
            plt.plot(x, cus, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("CUS_t")

            plt.subplot(616)
            plt.plot(x, cdg, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("CDG_t")

            plt.show()

    def baseline_local_opt_run(self, num_episode):
        pv = []
        pl = []
        e = []
        dg = []
        rt = []
        cus = []
        cdg = []
        totalReward_Avg = 0
        t_max = int(24/self.dt)
        days = 1
        # t_max = 2
        self.load_data(days)
        for i in range(num_episode):
            time, totalReward, done = 0, 0, False
            day = np.random.randint(0, days)
            state = self.reset(int(24/self.dt) * day)
            # for j in range(int(24/self.dt)):
            for j in range(t_max):
                r, a = self.myopic_policy(state)
                self.cus.pop()
                self.cdg.pop()
                state, r, _d, _ = self.step(a)  #derive state here
                r *= self.rewardScale
                totalReward += r
                dg.append(a)
                print("a:", a)
                cus.append(self.cus[j])
                print("CUS:", self.cus[j])
                cdg.append(self.cdg[j])
            print("Episode:",i,",score:",totalReward)
            totalReward_Avg = totalReward_Avg + totalReward

        totalReward_Avg = totalReward_Avg / num_episode
        print(" average score:",totalReward_Avg)

            # x = range(0, int(24/self.dt))
            # plt.subplot(611)
            # plt.plot(x, pv, marker='.', mec='r', mfc='w')
            # plt.xlabel("step")
            # plt.ylabel("PV")

            # plt.subplot(612)
            # plt.plot(x, pl, marker='.', mec='r', mfc='w')
            # plt.xlabel("step")
            # plt.ylabel("Load")

            # plt.subplot(613)
            # plt.plot(x, e, marker='.', mec='r', mfc='w')
            # plt.xlabel("step")
            # plt.ylabel("battery SOC")

            # plt.subplot(614)
            # plt.plot(x, dg, marker='.', mec='r', mfc='w')
            # plt.xlabel("step")
            # plt.ylabel("DG power")

            # plt.subplot(615)
            # plt.plot(x, cus, marker='.', mec='r', mfc='w')
            # plt.xlabel("step")
            # plt.ylabel("CUS_t")

            # plt.subplot(616)
            # plt.plot(x, cdg, marker='.', mec='r', mfc='w')
            # plt.xlabel("step")
            # plt.ylabel("CDG_t")

            # plt.show()

    def baseline_maxpower_run(self, num_episode):
        pv = []
        pl = []
        e = []
        dg = []
        rt = []
        cus = []
        cdg = []
        for i in range(num_episode):
            time, totalReward, done = 0, 0, False
            oPrime = self.reset()
            for j in range(int(24/self.dt)):
    
                a = self.dg_max
  
                oPrime, r, d, _ = self.step(a)
                r *= self.rewardScale
                totalReward += r
                pl.append(oPrime[0])
                pv.append(oPrime[1])
                e.append(oPrime[2])
                dg.append(a)
                cus.append(self.cus[j])
                cdg.append(self.cdg[j])
            print("Episode:",i,",score:",totalReward)
            x = range(0, int(24/self.dt))
            plt.subplot(611)
            plt.plot(x, pv, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("PV")

            plt.subplot(612)
            plt.plot(x, pl, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("Load")

            plt.subplot(613)
            plt.plot(x, e, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("battery SOC")

            plt.subplot(614)
            plt.plot(x, dg, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("DG power")

            plt.subplot(615)
            plt.plot(x, cus, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("CUS_t")

            plt.subplot(616)
            plt.plot(x, cdg, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("CDG_t")

            plt.show()




    def get_action_space_perstate(self):
        PE_ch_lim = self.PE_ch_lim
        PE_dis_lim = self.PE_dis_lim
        act_max = -self.state[1] + self.state[0] + PE_ch_lim
        act_min = -self.state[1] + self.state[0] - PE_dis_lim
        return act_max, act_min