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

ET_FAO_computed = ( 0.408*delta*(R_n-G) + psy*(e_0-e_a)*u*900/(T+273.15) ) / ( delta+ psy*(1+0.34*u) )
ET_priestley    = 1.26*(R_n-G)*delta/((delta+psy)*L)

def ET_blaney_criddle_func(K_0, K_1):
    coeffs = np.array([0.2100,0.2200,0.2300,0.2800,0.3000,0.3100,0.3000,0.2900,0.2700,0.2500,0.2200,0.2000])
    percentage = coeffs[T.index.month-1]
    return (T>0)*(K_1+K_0*(percentage*(0.46*T+8)-2))/24

from scipy.optimize import minimize
def MSE(X) -> float:
    sim_data = ET_blaney_criddle_func(*X)
    mse = np.mean( np.square(ET_FAO-sim_data) )
    print(X,mse)
    return mse

# bounds = [(0.1,None),(0.1,None)]
# X_opt = minimize(MSE, [1,1], method='L-BFGS-B', bounds=bounds).x
X_opt = [28.31876391, 18.78501231]
ET_FAO_computed   = ET_FAO_computed.resample("1D").mean()[1:-2]
ET_priestley      = ET_priestley.resample("1D").mean()[1:-2]
ET_blaney_criddle = ET_blaney_criddle_func(*X_opt).resample("1D").mean()[1:-2]


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


fig_plot, ax_plot                           = plt.subplots(1)
fig_scatter_FAO, ax_scatter_FAO             = plt.subplots(1)
fig_hist_FAO, ax_hist_FAO                   = plt.subplots(1)
fig_scatter_priestley, ax_scatter_priestley = plt.subplots(1)
fig_hist_priestley, ax_hist_priestley       = plt.subplots(1)
fig_scatter_blaney, ax_scatter_blaney       = plt.subplots(1)
fig_hist_blaney, ax_hist_blaney             = plt.subplots(1)

# [FAO] real/prediction
r_FAO = r2_score(ET_FAO, ET_FAO_computed)
ax_scatter_FAO.scatter(ET_FAO, ET_FAO_computed, color="green", s=0.2)
ax_scatter_FAO.plot([0,10],[0,10], color="black", linestyle="--")
ax_scatter_FAO.set_title(f"$R^2$: {r_FAO:0.3f}, RMSE: {np.sqrt(np.mean(np.square(ET_FAO-ET_FAO_computed))):0.5f}")
ax_scatter_FAO.set_aspect('equal', 'box') 
ax_scatter_FAO.set_xlim(-1,10)
ax_scatter_FAO.set_ylim(-1,10)
ax_scatter_FAO.set_ylabel("$y$")
ax_scatter_FAO.set_xlabel("$\hat{y}$")
# [FAO] hist real - prediction
ax_hist_FAO.hist(ET_FAO-ET_FAO_computed[:-2], bins=40, color="green")
ax_hist_FAO.set_title(r"FAO $y-\hat{y}$")

# [Priestley] real/prediction
r_priestley = r2_score(ET_FAO, ET_priestley)
ax_scatter_priestley.scatter(ET_FAO, ET_priestley, color="red", s=0.2)
ax_scatter_priestley.plot([0,10],[0,10], color="black", linestyle="--")
ax_scatter_priestley.set_title(f"$R^2$: {r_priestley:0.3f}, RMSE: {np.sqrt(np.mean(np.square(ET_FAO-ET_priestley))):0.2f}")
ax_scatter_priestley.set_aspect('equal', 'box') 
ax_scatter_priestley.set_xlim(-1,10)
ax_scatter_priestley.set_ylim(-1,10)
ax_scatter_priestley.set_ylabel("Priestley $y$")
ax_scatter_priestley.set_xlabel("$\hat{y}$")
# [Priestley] hist real - prediction
ax_hist_priestley.hist(ET_FAO-ET_priestley[:-2], bins=40, color="red")
ax_hist_priestley.set_title(r"Priestley $y-\hat{y}$")

# [Blaney Criddle] real/prediction
r_blaney = r2_score(ET_FAO, ET_blaney_criddle)
ax_scatter_blaney.scatter(ET_FAO, ET_blaney_criddle, color="green", s=0.2)
ax_scatter_blaney.plot([0,10],[0,10], color="black", linestyle="--")
ax_scatter_blaney.set_title(f"$R^2$: {r_blaney:0.3f}, RMSE: {np.sqrt(np.mean(np.square(ET_FAO-ET_blaney_criddle))):0.5f}")
ax_scatter_blaney.set_aspect('equal', 'box') 
ax_scatter_blaney.set_xlim(-1,10)
ax_scatter_blaney.set_ylim(-1,10)
ax_scatter_blaney.set_ylabel("Blaney $y$")
ax_scatter_blaney.set_xlabel("$\hat{y}$")
# [Blaney Criddle] hist real - prediction
ax_hist_blaney.hist(ET_FAO-ET_blaney_criddle[:-2], bins=40, color="green")
ax_hist_blaney.set_title(r"Blaney Criddle $y-\hat{y}$")

# [FAO] prediction
ax_plot.plot(ET_FAO_computed, color="green", linestyle="--", label="ET FAO")
# [Priestley] prediction
ax_plot.plot(ET_priestley, color="red", linestyle="--", label="ET priestley")
# [Blaney Criddle]
ax_plot.plot(ET_blaney_criddle, color="blue", linestyle="--", label="Blaney Criddle")
# [FAO from meteoschweiz]
ax_plot.plot(ET_FAO, color="black")
ax_plot.legend()

plt.show()

