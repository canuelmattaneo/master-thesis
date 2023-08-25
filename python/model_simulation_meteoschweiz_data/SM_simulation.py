'''
  Program to test the three evapotranspiration formulations with the one computed by Meteoschweiz
'''
import pandas as pd 
import numpy as np
import os
def path(filename):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, filename)

wanted_columns_1 = ["global_radiation [W/m^2]",
                    "shortwave_reflected_radiation [W/m^2]",
                    "longwave_outgoing_radiation [W/m^2]",
                    "longwave_incoming_radiation [W/m^2]",
                    "temperature_2m_above [C]",
                    "precipitation [mm]",
                    "relative_humidity_2m_above [%]",
                    "wind_speed [m/s]"] 
wanted_columns_2 = ["vapour_pressure_2m_above [hPa]",
                    "atmospheric_pressure [hPa]",
                    "evaporation [mm]"] 
new_columns = wanted_columns_1 + wanted_columns_2

dataset_1 = pd.read_csv(path("data_1.txt"), 
                         sep=';',
                         index_col="time", parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d%H"))

dataset_2 = pd.read_csv(path("data_2.txt"), 
                         sep=';',
                         index_col="time", parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d%H"))

dataset_3 = pd.read_csv(path("data_3.txt"), 
                        sep=';',
                        index_col="time", parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d"))

dataset = pd.DataFrame(columns=new_columns)
for col in new_columns:
    if col in wanted_columns_1:
        dataset[col] = dataset_1[col]
    if col in wanted_columns_2:
        dataset[col] = dataset_2[col]

t1,t2 = pd.to_datetime("2012-11-30"),pd.to_datetime("2023-04-15")
dataset_3 = dataset_3.loc[(dataset_3.index >= t1) & (dataset_3.index <= t2)]

ET_FAO = dataset_3["reference_evaporation_FAO [mm/d]"].astype(float)[1:]

glob    = dataset["global_radiation [W/m^2]"]
sw_refl = dataset["shortwave_reflected_radiation [W/m^2]"]
lw_out  = dataset["longwave_outgoing_radiation [W/m^2]"]
lw_in   = dataset["longwave_incoming_radiation [W/m^2]"]
P       = dataset["atmospheric_pressure [hPa]"]/10 # to kPa
T       = dataset["temperature_2m_above [C]"]
RH      = dataset["relative_humidity_2m_above [%]"]
u       = dataset["wind_speed [m/s]"]

L     = 0.0864*(28.4 - 0.028*T)                         # special heat of vaporization 
psy   = 0.665*1e-3*P                                    # psychrometric constant
e_0   = 0.61121*np.exp((18.678-T/234.5)*(T/(257.14+T))) # Buck eq. 
e_0   = 0.61094*np.exp(17.625*T/(T+243.04))             # Magnus eq. saturation vapour pressure at air temp.
e_a   = e_0*RH/100.0                                    # actual vapour pressure
delta = 4098*e_0 / np.square(T+273.15)                  # slope of vapour pressure curve

# R_n balance
R_n = glob - sw_refl + lw_in - lw_out
R_n = R_n * 0.0036 * 24               # conversion from (W/m^2)/h to (MJ/m^2)/h [1W=1J/s -> 1W*1sec = 1J]
# compute ground flux G
nighttime_mask         = (dataset.index.hour < 6) | (dataset.index.hour >= 18)
G                      = R_n*0.1
G.loc[nighttime_mask] *= 0.5/0.1

ET_0 = ( 0.408*delta*(R_n-G) + psy*(e_0-e_a)*u*900/(T+273.15) ) / ( delta+ psy*(1+0.34*u) )

W_0   = 0.3
W_max = 1500
psi   = 0.8

# Computation -----------------------------------------------------------------------
dt = 1.0
dataset["precipitation [mm]"][np.where(dataset["precipitation [mm]"]=="-")[0]] = np.nan
precipitation = dataset["precipitation [mm]"].astype(float).interpolate()
dataset["temperature_2m_above [C]"][np.where(dataset["temperature_2m_above [C]"]=="-")[0]] = np.nan
temperature = dataset["temperature_2m_above [C]"].astype(float).interpolate()

def compute(K_s_infiltration, K_c_evapotranspiration, K_s_drainage, drainage_exponent):
    ET_pot_vector = K_c_evapotranspiration*ET_0

    # init some variables
    W_prev            = W_0*W_max 
    rain_prev         = 0.0       # cumulative rain since last rainfall
    W_infiltration    = 0.0
    infiltration_flag = False
    W_init            = -np.inf   # water content at begin of a rainfall event
    F_ponding         = 0.0
    F                 = 1e-4

    N_sample = len(precipitation)
    W_vector              = np.zeros(N_sample)
    W_infiltration_vector = np.zeros(N_sample)
    W_ET_vector           = np.zeros(N_sample)


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
        W_ET_vector[n]           = W_ET
        #W_infiltration_vector[n] = W_infiltration
    return W_vector


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1) 

def calibrate(real_data, X_0):
    from scipy.optimize import minimize
    def RMSE(X) -> float:
        sim_data = compute(*X)
        MSE = np.sqrt( np.nanmean( np.square(real_data.values-sim_data) ) )

        print(X,MSE)
        return MSE
    
    bounds = [(0, None)] * len(X_0)
    X_opt = minimize(RMSE, X_0, method='L-BFGS-B', bounds=bounds).x

    return X_opt, RMSE(X_opt)


W_vector = compute(*[8, 0.1, 20, 13 ])
'''
W_vector = compute(*[15 , 1, 20, 25],
                   precipitation.interpolate(),
                   temperature.interpolate())

W_vector = compute(*[1.5, 1, 20, 4.5],
                   precipitation.interpolate(),
                   temperature.interpolate())


W_vector = compute(*[30.4243877 , 60.04019316, 39.10138902, 23.71072423],
                   precipitation.interpolate(),
                   temperature.interpolate())
'''

data2save = pd.DataFrame({"moisture":W_vector}, index=precipitation.index)
data2save["precipitation"] = precipitation.interpolate()
data2save["temperature"] = temperature.interpolate()
data2save.to_csv(path("sim_moisture.txt"))

