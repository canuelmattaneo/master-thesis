import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
def path(filename):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, filename)

ds = pd.read_csv(path("USDA-Bushland TX.txt"), index_col="date_time", parse_dates=True)
#ds = ds[len(ds)//2:]
#ds = ds[:len(ds)//2]

data = ds[["precipitation [mm]","dew [mm]","irrigation [mm]", "temperature [C]"]]
data.set_index(ds.index)

w_content = [261.0] # initial value
for idx,row in ds.iterrows():
    et = row["evapotranspiration [mm]"]
    i = row["irrigation [mm]"]
    p = row["precipitation [mm]"]
    d = row["dew [mm]"]

    w_content.append(w_content[-1] + i+p+d-et)

data["moisture [%]"] = np.array(w_content[1:])/450

moisture_real      = data["moisture [%]"]
precipitation      = data["precipitation [mm]"]+data["irrigation [mm]"]+data["dew [mm]"]
temperature        = data["temperature [C]"]

L=np.array([0.2100,0.2200,0.2300,0.2800,0.3000,0.3100,0.3000,0.2900,0.2700,0.2500,0.2200,0.2000])
percentage_daytime = 0.3#L[data.index.month]


W_0   = moisture_real[0]
W_max = 450
psi   = 0.8

# Computation -----------------------------------------------------------------------
N_sample = len(data)
dt = 0.5             # in hours

W_vector              = np.zeros(N_sample)
W_infiltration_vector = np.zeros(N_sample)
W_ET_vector           = np.zeros(N_sample)

K_s_infiltration       = 12.5
K_c_evapotranspiration = 1.24
K_s_drainage           = 18.85
drainage_exponent      = 22.69


def compute(K_s_infiltration, K_c_evapotranspiration, K_s_drainage, drainage_exponent):
    ET_pot_vector = (temperature>0).astype(np.int64) *\
                (K_c_evapotranspiration*(percentage_daytime *\
                (0.46*temperature+8.13)- 2))/(24/dt)

    # init some variables
    W_prev            = W_0*W_max 
    rain_prev         = 0.0       # cumulative rain since last rainfall
    W_infiltration    = 0.0
    infiltration_flag = False
    W_init            = -np.inf   # water content at begin of a rainfall event (Green Ampt.)
    F_ponding         = 0.0       # ponding water                              (Green Ampt.)
    F                 = 1e-4

    for n in range(N_sample):
        if n==3194:
            a=1
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
                        F=F+0.001


            ET = ET_pot_vector[n]*W/W_max
            W_ET = dt * ET

            drainage = K_s_drainage*(W/W_max)**(drainage_exponent)
            W_drainage = dt*drainage

            W_corr = W_prev + (W_infiltration - W_ET - W_drainage)

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

        W_vector[n] = W_corr/W_max

    return W_vector

def calibrate(real_data, X_0):
        
    def MSE(X) -> float:
        sim_data = compute(*X)
        MSE = np.nanmean( np.square(real_data-sim_data) )

        print(X,MSE)
        return MSE
    
    bounds = [(0.1, None)] * len(X_0)
    X_opt = minimize(MSE, X_0, method='L-BFGS-B', bounds=bounds).x

    return X_opt, MSE(X_opt)

#X_opt, mse = calibrate(moisture_real, [1,1.5,1,6])
#X_opt,mse = [1.19144376, 0.45841025, 0.42814363, 6.33558414], 0.00010673264795013801
# first half
#X_opt, mse = [1.19144376, 0.45841025, 0.42814363, 6.33558414], 0.00010673264795013801
#X_opt, mse = [5, 0.1, 41.0, 11.53 ], 0.00044
#X_opt, mse = [15, 0.1, 16.0, 9.65 ], 0.00044
# second half (normal data read and start by 261 and divide by 450 at the end)
X_opt, mse = [8, 0.1, 20, 13 ], 0.00044

W_vector = compute(*X_opt)

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import pandas as pd

fig,axs = plt.subplots(3)
print(np.mean(np.square(moisture_real-W_vector))**0.5)
axs[0].plot(moisture_real, label="Real moisture", linestyle="--")
axs[0].yaxis.set_major_locator(MultipleLocator(0.1))
axs[0].xaxis.set_major_locator(mdates.MonthLocator())
axs[0].set_ylim(0.3,0.8)    
axs[0].plot(moisture_real.index, W_vector, label="Fitted moisture")
axs[0].legend()
axs[0].grid()

axs[2].plot(precipitation, label="Precipitation")
axs[2].legend()
axs[2].grid()
axs[1].plot(temperature, label="Temperature")
axs[1].legend()
axs[1].grid()

axs[2].set_xlabel("Time")
plt.show()

