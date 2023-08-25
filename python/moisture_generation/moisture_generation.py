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
dataset_1 = pd.read_csv(path("data_1.txt"), 
                         sep=';',
                         index_col="time", 
                         parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d%H"))

wanted_columns_2 = ["vapour_pressure_2m_above [hPa]",
                    "atmospheric_pressure [hPa]",
                    "evaporation [mm]"] 
dataset_2 = pd.read_csv(path("data_2.txt"), 
                         sep=';',
                         index_col="time", 
                         parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d%H"))

new_columns = wanted_columns_1 + wanted_columns_2
dataset = pd.DataFrame(columns=new_columns)
for col in new_columns:
    if col in wanted_columns_1:
        dataset[col] = dataset_1[col]
    if col in wanted_columns_2:
        dataset[col] = dataset_2[col]

glob = dataset["global_radiation [W/m^2]"]
sw_refl = dataset["shortwave_reflected_radiation [W/m^2]"]
lw_out = dataset["longwave_outgoing_radiation [W/m^2]"]
lw_in = dataset["longwave_incoming_radiation [W/m^2]"]
P = dataset["atmospheric_pressure [hPa]"]/10 # to kPa
T = dataset["temperature_2m_above [C]"]
RH = dataset["relative_humidity_2m_above [%]"]
u = dataset["wind_speed [m/s]"]

L = 0.0864*(28.4 - 0.028*T) # special heat of vaporization 
psy = 0.665*1e-3*P # psychrometric constant
e_0 = 0.61121*np.exp((18.678-T/234.5)*(T/(257.14+T))) # <- Buck eq.  
e_0 = 0.61094*np.exp(17.625*T/(T+243.04)) # Magnus eq. saturation vapour pressure at air temp.
e_a = e_0*RH/100.0 # actual vapour pressure
delta = 4098*e_0 / np.square(T+273.15) # slope of vapour pressure curve
# R_n balance
R_n = glob - sw_refl + lw_in - lw_out
R_n = R_n * 0.0036*24 # conversion from (W/m^2)/h to (MJ/m^2)/h
# G
nighttime_mask = (dataset.index.hour < 6) | (dataset.index.hour >= 18)
G = R_n*0.1
G.loc[nighttime_mask] *= 0.5/0.1

ET_0 = ( 0.408*delta*(R_n-G) + psy*(e_0-e_a)*u*900/(T+273.15) ) / ( delta+ psy*(1+0.34*u) )
ET_priestley = 1.26*(R_n-G)*delta/((delta+psy)*L)


ET_FAO_computed = ET_0.resample("1D").mean()[1:-2]
ET_priestley = ET_priestley.resample("1D").mean()[1:-2]

precipitation      = dataset["precipitation [mm]"].replace('-', np.nan).astype(float)
temperature        = dataset["temperature_2m_above [C]"].replace('-', np.nan).astype(float)
percentage_daytime = 0.3*np.ones(len(temperature))

# Computation -----------------------------------------------------------------------

def compute(K_s_infiltration, K_c_evapotranspiration, K_s_drainage, drainage_exponent, precipitation, temperature):
    N_sample = len(precipitation)
    dt = 0.25
    ET_pot_vector = (temperature>0).astype(np.int64) *\
                (K_c_evapotranspiration*(0.3 *\
                (0.46*temperature+8.13)- 2))/(24/dt)
    
    W_vector              = np.zeros(N_sample)
    W_infiltration_vector = np.zeros(N_sample)
    W_ET_vector           = np.zeros(N_sample)
    W_0   = 0.3
    W_max = 4000
    psi   = 0.8

    
    #ET_pot_vector = K_c_evapotranspiration*ET_0

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



N_sample = len(precipitation)
dt = 0.25

W_max = 600
W_vector              = np.zeros(N_sample)
W_vector_rate         = np.zeros(N_sample)
W_infiltration_vector = np.zeros(N_sample)
W_ET_vector           = np.zeros(N_sample)

K_s_infiltration       = 12.5
K_c_evapotranspiration = 1.24
K_s_drainage           = 18.85
drainage_exponent      = 22.69


def compute(K_s_infiltration, K_c_evapotranspiration, K_s_drainage, drainage_exponent, precipitation, temperature):
    ET_pot_vector = (temperature>0).astype(np.int64) *\
                (K_c_evapotranspiration*(0.3 *\
                (0.46*temperature+8.13)- 2))/(24/dt)

    # init some variables
    psi = 0.8
    W_0 = 0.8
    W_prev            = W_0*W_max 
    rain_prev         = 0.0       # cumulative rain since last rainfall
    W_infiltration    = 0.0
    infiltration_flag = False
    W_init            = -np.inf   # water content at begin of a rainfall event
    F_ponding         = 0.0
    F                 = 1e-4

    for n in range(len(precipitation)):
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

            W_corr = W_prev + (W_infiltration - W_ET - W_drainage)
            W_corr_rate = (W_infiltration - W_ET - W_drainage)/dt

            if W_corr>=W_max:
                W_corr=W_max
            
            W_err = np.abs((W_corr-W)/W_max)
            W = W_corr
            if W_iteration==100:
                break
            W_iteration += 1

        W_prev=W_corr
        if n>2:
            rain_prev=np.sum(precipitation[n-3:n+1])

        W_vector[n]              = W_corr/W_max
        W_vector_rate[n]         = W_corr_rate/W_max
        #W_ET_vector[n]           = W_ET
        #W_infiltration_vector[n] = W_infiltration
    return W_vector, W_vector_rate


show=False

if show:
    dataset = dataset[pd.to_datetime("2013-09-01"):pd.to_datetime("2015-01-01")]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    
    W_vector_1 = compute(12.4243877 , 10.04019316, 19.10138902, 23.71072423,
                    precipitation.interpolate(), 
                    temperature.interpolate())

    W_vector_2 = compute(1.9979564,  1., 2.25002305,  8.49158866, 
                    precipitation.interpolate(), 
                    temperature.interpolate())

    W_vector_3 = compute(20., 1., 1.59832158, 7.77347634,
                    precipitation.interpolate(), 
                    temperature.interpolate())


    ax.plot(precipitation.index, W_vector_1)
    ax.plot(precipitation.index, W_vector_2)
    ax.plot(precipitation.index, W_vector_3)

    plt.show()

else:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2)
    
    import statsmodels.api as sm

    #temperature = temperature[pd.to_datetime("2013-09-01"):pd.to_datetime("2018-01-01")]
    #precipitation = precipitation[pd.to_datetime("2013-09-01"):pd.to_datetime("2018-01-01")]
    #precipitation[pd.to_datetime("2014-03-01"):pd.to_datetime("2014-04-01")] = 0
    W_vector,W_rate = compute(*[30.4243877 , 60.04019316, 39.10138902, 23.71072423], 
                      precipitation, 
                      temperature.interpolate())
    W_vector = W_vector[:len(temperature)]
    W_rate = W_rate[:len(temperature)]

    #K_s_infiltration, K_c_evapotranspiration, K_s_drainage, drainage_exponent

    ax[0].plot(precipitation.index, W_vector)
    ax[0].twinx().plot(precipitation.index,W_rate, color="green")
    #ax[0].twinx().plot(precipitation.index, precipitation,color="green")
    ax[0].set_ylim(0.1,1.1)
    #ax[0].twinx().plot(precipitation, color="red")
    ax[-1].plot(temperature)
    plt.show()

    
