import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import h5py
import random
# use the standard SIR model to generate sample data for training
add_noise = True # False

def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + x0))) + R_0_end

#with h5py.File("../raw-data/sampled/sir_training_rk.hdf5", "w") as f:
S, I, R = [], [], []
for j in range(7):
    for i in range(2):
        k, b = random.random(), random.random()

        def Model(days, k, b): # agegroups,
            S0 = random.random()
            I0 = random.random() * (1 - S0)
            R0 = 1 - S0- I0
            y0 = S0 * 100, I0 * 100, R0 * 100 # S0, I0, R0
            t = np.linspace(0, days, days)

            ret = odeint(deriv, y0, t, args=(k, b))
            # Save to HDF5
            ret_df = pd.DataFrame(ret)
            #dset = f.create_dataset(str(i), data=ret_df)

            S_i, I_i, R_i = ret.T
            S.extend(S_i)
            I.extend(I_i)
            R.extend(R_i)

            return t, S, I, R

        def deriv(y, t, k, b):
            S, I, R = y

            dSdt = -b * I * S/100

            dRdt = k * I

            dIdt = -(dSdt + dRdt)

            return dSdt, dIdt, dRdt

        t, S, I, R = Model(257, 1/3, 0.5)#k, b) # 2048, 256,

    df = pd.DataFrame(np.array([S, I, R]).T)
    df.to_csv("../SIR_train_"+str(j)+".csv", index=False)

'''plt.plot(t, S, label='S')
        plt.plot(t, I, label='I')
        plt.plot(t, R, label='R')
        plt.title("b="+str(b)+"k="+ str(k))
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.show()'''



#S_values,  = [S(t) for x in t]
''' Original code
gamma = 1.0 / 9.0
sigma = 1.0 / 3.0

def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + x0))) + R_0_end


def Model(days, agegroups, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
    def beta(t):
        return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma

    N = sum(agegroups)

    def Beds(t):
        beds_0 = beds_per_100k / 100_000 * N
        return beds_0 + s * beds_0 * t  # 0.003

    y0 = N - 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    t = np.linspace(0, days, days)
    ret = odeint(deriv, y0, t, args=(R_0_start, k, x0, R_0_end, gamma, sigma, N, prob_I_to_C, prob_C_to_D, Beds))
    S, E, I, C, R, D = ret.T

    def betaa(t):
        return I[t] / (I[t] + C[t]) * (12 * prob_I_to_C + 1 / gamma * (1 - prob_I_to_C)) + C[t] / (I[t] + C[t]) * (
                min(Beds(t), C[t]) / (min(Beds(t), C[t]) + max(0, C[t] - Beds(t))) * (
                    prob_C_to_D * 7.5 + (1 - prob_C_to_D) * 6.5) +
                max(0, C[t] - Beds(t)) / (min(Beds(t), C[t]) + max(0, C[t] - Beds(t))) * 1 * 1
        )

    def beta(t):
        return logistic_R_0(t, R_0_start, k, x0, R_0_end) / betaa(t) if not np.isnan(betaa(t)) else 0

    R_0_over_time = [beta(i) / gamma for i in range(len(t))]

    return t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D
def deriv(y, t, R_0_start, k, x0, R_0_end, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
    S, E, I, C, R, D = y
    def betaa(t):
        return I / (I + C) * (12 * p_I_to_C + 1/gamma * (1 - p_I_to_C)) + C / (I + C) * (
                    min(Beds(t), C) / (min(Beds(t), C) + max(0, C-Beds(t))) * (p_C_to_D * 7.5 + (1 - p_C_to_D) * 6.5) +
                    max(0, C-Beds(t)) / (min(Beds(t), C) + max(0, C-Beds(t))) * 1 * 1
                             )
    def beta(t):
        return logistic_R_0(t, R_0_start, k, x0, R_0_end) / betaa(t) if not np.isnan(betaa(t)) else 0

    dSdt = -beta(t) * I * S / N
    dEdt = beta(t) * I * S / N - sigma * E
    dIdt = sigma * E - 1/12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    dCdt = 1/12.0 * p_I_to_C * I - 1/7.5 * p_C_to_D * min(Beds(t), C) - max(0, C-Beds(t)) - (1 - p_C_to_D) * 1/6.5 * min(Beds(t), C)
    dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1/6.5 * min(Beds(t), C)
    dDdt = 1/7.5 * p_C_to_D * min(Beds(t), C) + max(0, C-Beds(t))
    return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt
    '''