import numpy as np
from scipy.optimize import minimize, differential_evolution
import os
import pandas as pd
def path(filename):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, filename)

data = pd.read_csv(path("Paper L.Brocca.txt"))

moisture_real = data["moisture [%]"]
rain          = data["precipitation [mm]"]
T             = data["temperature [C]"]

data=data.set_index(pd.date_range(start="10.09.2000 00:00", periods=22996, freq="30T"))#[:11000]

moisture_real    = data["moisture [%]"]
rain      = data["precipitation [mm]"]
T   = data["temperature [C]"]

N = len(moisture_real)
n = np.arange(N)

#offset = (n/17e3)**2
#offset = 2*np.ones(N)
def poly(t,coeffs):
    return np.sum([coeff*t**n for n,coeff in enumerate(coeffs)], axis=0)
t      = np.linspace(0,250,N)
coeffs = np.array([0, 1.91868270e-01, -1.92342775e-04,  2.06333069e-05])/4
tree   = poly(t, coeffs)

from scipy.interpolate import CubicSpline
# random spline perturbation function
n_ctrl_pts = 10
x_ctrl = np.linspace(t[0], t[-1], n_ctrl_pts)
y_ctrl = np.random.random(size=n_ctrl_pts) # range -> [0,1)
y_ctrl[0] = y_ctrl[-1] = 0 # fix boundary to avoid a shift, i.e change in the function
spline = CubicSpline(x_ctrl, y_ctrl*10)
offset = (tree + spline(t)/2)/30 #+ np.random.randn(N)

z = offset+moisture_real.values + np.random.normal(size=len(offset))/50
#z = pd.DataFrame(z).interpolate().values
dhat   = np.zeros(N)
states = np.zeros((2,N))*np.nan
Xhat   = np.array([0.5, 2])
P      = np.eye(2)*1
Qv     = 0.5*np.nanvar(z)
Qw     = np.array([[1, 0],
                   [0, 1]]) 

# np.diff(y_sim) -> is not 0 (should be) -> error of tree model -> statistics


# Computation -----------------------------------------------------------------------
M = len(data)
dt = 0.5

# Model parameter
W_0_RZ   = 0.5184
W_max    = 60.5573 # mm
Psi_av_L = -0.8

Ks_sup   = 12.5050 # mm/h
Ks_RZ    = 18.8511 # mm/h
m_RZ     = 22.6978
Kc       = 1.2488

# Parameters adjustment
Ks_sup=Ks_sup*dt
Ks_RZ=Ks_RZ*dt

# Potential Evapotranspiration parameter
L=np.array([0.2100,0.2200,0.2300,0.2800,0.3000,0.3100,0.3000,0.2900,0.2700,0.2500,0.2200,0.2000])
Ka=1.26
EPOT=(T>0)*(Kc*(Ka*L[data.index.month-1]*(0.46*T+8)-2))/(24/dt)


# Thresholds for numerical computations
soglia=0.01  # Water Content
soglia1=0.01 # Infiltration


nan_mask = np.isnan(moisture_real.values)

count=0
for min,max in [(0,np.where(nan_mask==True)[0].min()),(np.where(nan_mask==True)[0].max()+1,N)]:
    
    # Parameter initialization
    F_p=0
    F=0.00001
    W_init=-9999
    WW=np.zeros(M)
    W_RZ_p=W_0_RZ*W_max
    rainprec=0
    inf=0
    ii=1 
    i=1


    W_0_RZ = [0.518, 0.088][count]
    t_0    = [0.0  , z[min]][count]
    W_RZ_p=W_0_RZ*W_max
    Xhat = np.array([W_0_RZ, t_0])
    count+=1

    for t in range(min,max):
        # W_{n+1} = f(W_n)
        W_RZ=W_RZ_p
        jj=0
        err=1000
        while err>soglia:
            # infiltration computation (Green-Ampt)
            jj=jj+1
            if (rainprec<=0.001):
                if (rain[t]>0):
                    W_init=W_RZ
                    Psi=Psi_av_L*(W_max-W_init)
                else:
                    W_init=-9999
                    ii=1
                    F_p=0
                    F=0.00001

            if W_init != -9999:
                if jj>1:
                    ii=ii-1
                
                if ii==1:
                    inf=1000
                else:
                    inf = Ks_sup*(1-Psi/F)

                if inf>rain[t]:
                    inf=rain[t]
                    F=F+rain[t]
                else:
                    F_p=F
                    errore=1000
                    while errore>soglia1: # iteration for the infiltration
                        F1=F_p-Psi*np.log((F-Psi)/(F_p-Psi))+Ks_sup
                        errore=abs((F-F1)/F)
                        F=F+0.01
                    
                    inf=F-F_p
                
                F_p=F
                ii=ii+1

            e         = EPOT[t]*W_RZ/W_max
            perc      = Ks_RZ*(W_RZ/W_max)**(m_RZ)
            W_RZ_corr = W_RZ_p+(inf-e-perc)

            if W_RZ_corr>=W_max:
                W_RZ_corr=W_max
            

            err=np.abs((W_RZ_corr-W_RZ)/W_max)
            W_RZ=W_RZ_corr
            if jj==100:
                break
        
        W_RZ_p=W_RZ_corr # W_RZ_p = W_{n+1}
        if t>3:
            rainprec=sum(rain[t-3:t+1])

        # W_n [%] = W_RZ_corr/W_max 
        WW[t] = W_RZ_corr/W_max
        W     = W_RZ_corr/W_max
        dWdt  = (inf-e-perc)/dt

        '''
        Extended Kalman filter
        '''
        A = np.array([[dWdt, 0],
                      [0,    1]])
        '''
        w_n = f(w_{n-1}) <- nicht linear
        t_n = t_{n-1}
        
        - [ALMOST DONE] messungsprozess simulieren (noise + quantisierung + see datasheet for resolution all signals)
        - [DONE] new data to asses statistics
          -> simulation + real
        - [NOT DONE] best tree model (green mass oscillation)
        - [DONE] decomposition model vs simulation model (which one is it worth to improve)
        - [DONE] inputs in the kalman decomposition model are weather data (because it uses the simulation)
          -> make measurements worse and compare results (kalman: weather 
                                                          LMS   : water content sensor ... )
             what are pro and cons. of all models

             x-axis -> how bad the noise/measurements is (do we need good sensors?)
             y-axis -> ex. MSE

        - concept of virtual sensors 
          -> use model of evaporation
        '''
        H = np.array([1, 1])
        Xhatminus   = np.array([W, Xhat[1]])
        Pminus      = A@P@A.T + Qw
        K           = (Pminus@H.T) / (H@Pminus@H.T + Qv) 
        Xhat        = Xhatminus + K*(z[t]-H@Xhatminus)
        P           = (np.eye(2)-np.outer(K,H)) @ Pminus 
        dhat[t]     = H@Xhat
        states[:,t] = Xhat
        W_RZ_p      = W_max*(Xhat[0] if Xhat[0] < 1.0 else 1.0)
        '''
        end Extended Kalman filter
        '''
        
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.ticker import MultipleLocator
    import matplotlib.dates as mdates

    fig,ax = plt.subplots(1)

    date_form = mdates.DateFormatter("%m/%d/%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    ax.tick_params(axis='x', rotation=45)
    


    ax.plot(data.index, z, color="darkblue", label = "$f_T[n]+f_W[n]$")
    ax.plot(data.index, states[0,:]+states[1,:], color="blue", linestyle="--")

    ax.plot(data.index, moisture_real, color="grey", linestyle='--')
    ax.plot(data.index, states[0,:], color="black", label="estimation of $f_W[n]$, $\hat{f_W}[n]$")
    
    ax.plot(data.index, offset, color="green", linestyle='--')
    ax.plot(data.index, pd.DataFrame(states[1,:]).rolling(200).mean(), color="darkgreen", label="estimation of $f_T[n]$, $\hat{f_T}[n]$")

    ax.legend()
    
    print(f"MSE t[n]+w[n] : {np.nanmean(np.square(z      - (states[0,:]+states[1,:])))**0.5}")
    print(f"MSE t[n]      : {np.nanmean(np.square(offset - states[1,:]))**0.5}")
    print(f"MSE w[n]      : {np.nanmean(np.square(moisture_real  - states[0,:]))**0.5}")
    
    plt.show()



    '''
    
    [validation model]
    f(coeffs, meteo_data) -> water_sim (validation with real data)
    -> water_content = f(..., meteo_data) (extended kalman)
    
    ...[validation kalman aproach]

    bis 26 Sitzungen
    26.07-21.08 : Ferien 
    23 -> abgabe
    '''