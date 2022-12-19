# environment for microgrid


import numpy as np
import time
import random
import matplotlib.pyplot as plt

class env_mg:
    def __init__(self,):
        
        self.dg_max=850 # DG最大发电功率，单位kW
        self.dg_min=100 # DG最小发电功率，单位kW
        self.bdg_max=1 # DG开关打开的概率最大值
        self.bdg_min=0 # DG开关打开的概率最小值
        self.action_max = np.array([self.dg_max, self.bdg_max])
        self.action_min = np.array([self.dg_min, self.bdg_min])
        self.dt=1/4 # 时间颗粒为1/4h=15min
        self.PE_max=60 # 电池最大功率，单位kW
        self.E_max=600 # 电池最大电量，单位kWh
        self.E_min=6 # 电池最小电量，单位kWh
        self.eta_ch=0.98 # charging efficiency
        self.eta_dis=0.98 # discharging efficiency
        self.PV_max=300 # PV最大输出功率
        self.PV_min=0 # PV最小输出功率
        self.load_max= 1500 # load最大功率
        self.load_min= 10 # load最小功率
        self.actionDim=2 # 动作空间维数
        self.stateDim=4 # 状态空间维数
        self.hisStep=4 # history使用的历史帧数
        self.C_SD=1 # a fixed start-up cost
        self.C_RD=100 # a fixed running cost
        self.cus=[]
        self.cdg=[]
        self.state=[]
        self.obs=[]
        self.PE_ch_lim=self.PE_max
        self.PE_dis_lim = self.PE_max
        # self.rewardScale = 2e-5

    def step(self,u):
        # u是动作，u[0] stands for active power output of DG；u[1], 1 stands for change, while 0 stands for remains unchanged.
        dt = self.dt
        PL_t = self.state[0]
        PPV_t = self.state[1]


        # CSDG_t, a fixed start-up cost
        if u[1] > 0.5:
            u[1] = 1
        else:
            u[1] = 0

        # 开始计算costs
        ad = 0.005
        bd = 6
        cd = 100

        if u[1] == 1:
            CDG_t = (ad * u[0] ** 2 + bd * u[0] + cd) * dt
        else:
            CDG_t=0
        self.cdg.append(CDG_t)

        B_DG_t = self.B_DG_t
        CSDG_t = self.C_SD * u[1] * (1 - B_DG_t)

        # CRDG_t, a fixed running cost
        if B_DG_t == 0:
            B_DG_t1 = u[1]
        else:
            B_DG_t1 = 1 - u[1]
        self.B_DG_t = B_DG_t1
        CRDG_t = self.C_RD * B_DG_t1

        # CUS_t
        E_t = self.E_t # E_t_1是E_t-1，E_t1是E_t+1
        PE_ch_lim = min(self.PE_max, (self.E_max-E_t)/self.eta_ch/dt)
        PE_dis_lim = min(self.PE_max, self.eta_dis*(E_t-self.E_min)/dt)
        self.PE_ch_lim = PE_ch_lim
        self.PE_dis_lim = PE_dis_lim
        delta_t = u[0]*u[1] + PPV_t - PL_t
        k1 = 100
        k2 = 100
        if delta_t > PE_ch_lim:
            CUS_t = k1 * (delta_t - PE_ch_lim) * dtENV
            self.cus.append(CUS_t / k1)
        elif delta_t < -PE_dis_lim:
            CUS_t = -k2 * (delta_t - PE_dis_lim) * dt
            self.cus.append(-CUS_t / k2)
        else:
            CUS_t = 0
            self.cus.append(0)
        costs = CDG_t + CUS_t + CSDG_t + CRDG_t
        #costs = CUS_t
        
        # costs报错机制
        if costs < 0:
            raise Exception("costs equals to ",costs, ", costs should not be less than 0!")

        if delta_t >= 0:
            u_t = 1
            PE_t = min(delta_t,PE_ch_lim)
        else:
            u_t = 0
            PE_t = min(-delta_t,PE_dis_lim)
        E_t1 = E_t + self.eta_ch * u_t * PE_t * dt - (1-u_t) * PE_t * dt / self.eta_dis
        self.E_t = E_t1
        # 读两行数据
        f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        lines = f.readlines()
        cnt = self.cnt
        PL_t = float(lines[cnt].strip().split(',')[-2])
        PPV_t = float(lines[cnt].strip().split(',')[-1])
        PL_t_1 = float(lines[cnt-1].strip().split(',')[-2])
        PPV_t_1 = float(lines[cnt-1].strip().split(',')[-1])
        self.cnt += 1
        # 得到state，PL_t , PPV_t从历史数据中读出，E_t由公式算出
        state = np.array([PL_t , PPV_t , E_t1, B_DG_t1])
        self.state = state
        # 得到observations，PL_t_1 , PPV_t_1从历史数据中读出，E_t由公式算出
        obs = np.array([PL_t_1 , PPV_t_1 , E_t1, B_DG_t1])
        self.obs = obs
        # # print("state=",state,",observation=",obs,",reward=",-costs)
        return state, -costs, False, {}

    def reset(self):
        f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        lines=f.readlines()
        # np.random.seed(int(time.time()))
        # np.random.seed(123)
        # n=np.random.random_integers(1,len(lines)-24/self.dt) # 523866是dt=1min的数据总行数，523866/(60*1/4)是dt=15min的数据总行数
        # n=10519
        # n=128
        n = 24900 # 针i对local算法，reset()的返回值也要改成obs
#        n = 24900 - self.hisStep # 针对RDPG算法
        # print('Start from line ' + str(n) + ' in file data_dt_' + str(int(60 * self.dt)) + '.csv')
        self.cnt=n
        PL_t = float(lines[n].strip().split(',')[-2])
        PPV_t = float(lines[n].strip().split(',')[-1])
        PL_t_1 = float(lines[n - 1].strip().split(',')[-2])
        PPV_t_1 = float(lines[n - 1].strip().split(',')[-1])
        self.cnt += 1
        # self.E_t = np.random.random_integers(self.E_min, self.E_max)
        self.E_t = self.E_min
        # self.B_DG_t = random.random() # on/off status of DG_d, where 1 stands for on, while 0 stands for off.
        # if self.B_DG_t > 0.5:
        #     self.B_DG_t = 1 # on/off status of DG_d, where 1 stands for on, while 0 stands for off.
        # else:
        #     self.B_DG_t = 0
        self.B_DG_t = 1
        state = np.array([PL_t, PPV_t, self.E_t, self.B_DG_t])
        self.state = state
        # return state
        obs = np.array([PL_t_1, PPV_t_1, self.E_t, self.B_DG_t])
        self.obs = obs
        return state

    def run(self, num_episode):
        f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        lines = f.readlines()
        n = np.random.random_integers(1,len(lines) - 24 / self.dt)  # 523866是dt=1min的数据总行数，523866/(60*1/4)是dt=15min的数据总行数
        # print('Start from line'+str(n)+' in file data_dt_' + str(int(60 * self.dt)) + '.csv')
        self.cnt = n
        o = self.reset()
        pv = []
        pl = []
        e = []
        dg = []
        rt = []
        for i in range(num_episode):
            totalReward = 0
            for j in range(int(24/self.dt)):
                np.random.seed(int(time.time()+j))
                dg = np.random.random_integers(self.action_min, self.action_max)
                oPrime, r, d, _ = self.step(a)
                totalReward += r
                pl.append(oPrime[0])
                pv.append(oPrime[1])
                e.append(oPrime[2])
                dg.append(a)
                rt.append(r)
            print("Episode:",i,",score:",totalReward)
            x = range(0, int(24/self.dt))
            plt.subplot(511)
            plt.plot(x, pv, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("PV")

            plt.subplot(512)
            plt.plot(x, pl, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("Load")

            plt.subplot(513)
            plt.plot(x, e, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("battery SOC")

            plt.subplot(514)
            plt.plot(x, dg, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("DG power")

            plt.subplot(515)
            plt.plot(x, rt, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("reward")

            plt.show()

    def baseline_local_opt_run(self, num_episode):
        f = open('./data/data_dt_' + str(int(60 * self.dt)) + '.csv', 'r')
        lines = f.readlines()
    #    n=28
    #    n = np.random.random_integers(1,len(lines) - 24 / self.dt)  # 523866是dt=1min的数据总行数，523866/(60*1/4)是dt=15min的数据总行数
        # print('Start from line'+str(n)+' in file data_dt_' + str(int(60 * self.dt)) + '.csv')
    #    self.cnt = n
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
                np.random.seed(j)
                PL = oPrime[0]
                PV = oPrime[1]
                if PL - PV > self.dg_max:
                    a = self.dg_max
                elif PL - PV <= self.dg_max and PL - PV >= self.dg_min:
                    a = PL - PV
                else:
                    a = self.dg_min
                # a = np.random.random_integers(self.dg_min, self.dg_max)
                # a = self.dg_max
                oPrime, r, d, _ = self.step(a)
                r *= 2e-5
                totalReward += r
                pl.append(oPrime[0])
                pv.append(oPrime[1])
                e.append(oPrime[2])
                dg.append(a)
                rt.append(r)
                cus.append(self.cus[j])
                cdg.append(self.cdg[j])
            print("Episode:",i,",score:",totalReward)
            x = range(0, int(24/self.dt))
            plt.subplot(711)
            plt.plot(x, pv, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("PV")

            plt.subplot(712)
            plt.plot(x, pl, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("Load")

            plt.subplot(713)
            plt.plot(x, e, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("battery SOC")

            plt.subplot(714)
            plt.plot(x, dg, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("DG power")

            plt.subplot(715)
            plt.plot(x, rt, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("reward")

            plt.subplot(716)
            plt.plot(x, cus, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("CUS_t(without k)")

            plt.subplot(717)
            plt.plot(x, cdg, marker='.', mec='r', mfc='w')
            plt.xlabel("step")
            plt.ylabel("CDG_t")

            plt.show()