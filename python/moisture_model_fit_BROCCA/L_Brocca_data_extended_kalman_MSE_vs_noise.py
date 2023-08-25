import numpy as np
from scipy.optimize import minimize, differential_evolution
import os
import pandas as pd
def path(filename):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, filename)

data = pd.read_csv(path("Paper L.Brocca.txt"))
data=data.set_index(pd.date_range(start="10.09.2000 00:00", periods=22996, freq="30T"))#[:11000]

moisture_real = data["moisture [%]"]
precipitation = data["precipitation [mm]"]
T             = data["temperature [C]"]

percentage_daytime = 0.3*np.ones(len(T))

N = len(moisture_real)
n = np.arange(N)

offset = (n/17e3)**2
#offset = 2*np.ones(N)

var_vector = [1,5,10,15,20,25,30,35,40,45,50]
MSE_vector = []

for var in var_vector:
    z = offset+moisture_real.values + np.random.normal(size=len(offset))/var
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
        PIOprec=0
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
                if (PIOprec<=0.001):
                    if (precipitation[t]>0):
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

                    if inf>precipitation[t]:
                        inf=precipitation[t]
                        F=F+precipitation[t]
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
                PIOprec=sum(precipitation[t-3:t+1])

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
            
            - messungsprozess simulieren (noise + quantisierung + see datasheet for resolution all signals)
            - new data to asses statistics
            -> simulation + real
            - best tree model (green mass oscillation)
            - decomposition model vs simulation model (which one is it worth to improve)
            - inputs in the kalman decomposition model are weather data (because it uses the simulation)
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
            
    MSE_vector.append(np.nanmean(np.square(z-(states[0,:]+states[1,:])))**0.5)
    print(MSE_vector[-1])

print(MSE_vector)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    MSE_vector = [0.23831710723124486, 0.019534848865840372, 0.008719299468883399, 0.0061118881422292425, 0.004601487564154671, 0.003680580049366164, 0.0030625281151711063, 0.0026227675933740333, 0.002302490289539712, 0.002046157622472415, 0.0018366477373652325]
    var_vector = [1,5,10,15,20,25,30,35,40,45,50]
    plt.plot(var_vector,MSE_vector[::-1])    
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