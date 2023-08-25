import numpy as np
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import os
def path(filename):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, filename)

data = pd.read_csv(path("Paper L.Brocca.txt"))

moisture_real      = data["moisture [%]"]
precipitation      = data["precipitation [mm]"]
temperature        = data["temperature [C]"]
percentage_daytime = 0.3*np.ones(len(temperature))

W_0   = moisture_real[0]
W_max = 60
psi   = 0.8

# Computation -----------------------------------------------------------------------
N_sample = len(precipitation)
dt = 0.25

W_vector              = np.zeros(N_sample)
W_infiltration_vector = np.zeros(N_sample)
W_ET_vector           = np.zeros(N_sample)

K_s_infiltration       = 12.5
K_c_evapotranspiration = 1.24
K_s_drainage           = 18.85
drainage_exponent      = 22.69


def compute(K_s_infiltration, K_c_evapotranspiration, K_s_drainage, drainage_exponent, precipitation):
    ET_pot_vector = (temperature>0).astype(np.int64) *\
                (K_c_evapotranspiration*(percentage_daytime *\
                (0.46*temperature+8.13)- 2))/(24/dt)

    # init some variables
    W_prev            = W_0*W_max 
    rain_prev         = 0.0       # cumulative rain since last rainfall
    W_infiltration    = 0.0
    infiltration_flag = False
    W_init            = -np.inf   # water content at begin of a rainfall event
    F_ponding         = 0.0
    F                 = 1e-4

    for n in range(N_sample):
        W = W_prev
        W_iteration = 0
        epsilon_W = epsilon_F = 1e-3
        W_err = epsilon_W+1
        while W_err > epsilon_W:

            if rain_prev <= 1e-3:
                if precipitation[n]>0:
                    W_init = W
                    infiltration_flag = True
                else:
                    infiltration_flag = False
                    F_ponding=0
                    F=1e-4

            if infiltration_flag:
                W_infiltration = dt*K_s_infiltration*( 1 + psi*(W_max-W_init)/F )

                if W_infiltration > precipitation[n]:
                    W_infiltration = precipitation[n]

                    if W_iteration==0:
                        F += precipitation[n]
                    
                else:
                    F_ponding = F
                    F_err = epsilon_F+1
                    while F_err > epsilon_F:
                        F_next = F_ponding + psi*np.log( (F+psi) / (F_ponding+psi) ) + K_s_infiltration*dt
                        F_err = np.abs( (F-F_next)/F )
                        F=F+0.01


            ET = ET_pot_vector[n]*W/W_max

            W_ET = dt * ET

            drainage = K_s_drainage*(W/W_max)**(drainage_exponent)
            W_drainage = dt*drainage

            W_corr = W_prev + (W_infiltration - W_ET - W_drainage);

            if W_corr>=W_max:
                W_corr=W_max
            
            W_err = np.abs((W_corr-W)/W_max);
            W = W_corr
            if W_iteration==100:
                break
            W_iteration += 1

        W_prev=W_corr
        if n>2:
            rain_prev=np.sum(precipitation[n-3:n+1])

        W_vector[n]              = W_corr/W_max
        #W_ET_vector[n]           = W_ET
        #W_infiltration_vector[n] = W_infiltration
    return W_vector

def calibrate(real_data, X_0):
        
    def MSE(X) -> float:
        sim_data = compute(*X)
        MSE = np.nanmean( np.square(real_data-sim_data) )

        print(X,MSE)
        return MSE
    
    bounds = [(0, None)] * len(X_0)
    X_opt = minimize(MSE, X_0, method='L-BFGS-B', bounds=bounds).x

    return X_opt, MSE(X_opt)

#print(calibrate(moisture_real, np.random.randn(4)))
W_vector = compute(*[12.4243877 , 10.04019316, 19.10138902, 23.71072423], precipitation)
#[ 7.82957264 10.14222333 72.35550491 30.90686849] 0.004587714252831221]
print(np.sqrt(np.nanmean( np.square(moisture_real.values-W_vector) )))
import matplotlib.pyplot as plt
from random import shuffle
import pandas as pd


print(np.mean(np.square(moisture_real-W_vector))**0.5)
fig,axs = plt.subplots(3)
axs[0].set_ylim(0,1)
axs[0].set_ylim(0,1)
axs[0].plot(moisture_real, linestyle="--", label="Real moisture")
axs[0].plot(moisture_real.index,W_vector, label="Fitted moisture")
axs[0].legend()
axs[0].grid()
axs[1].plot(moisture_real.index,temperature, label="Temperature")
axs[1].legend()
axs[1].grid()
axs[2].plot(moisture_real.index, precipitation.values, label="Precipitation")
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel("Time")

plt.show()