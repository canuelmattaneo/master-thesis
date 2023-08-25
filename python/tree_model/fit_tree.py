import pandas as pd
import os
def path(filename):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, filename)

ds = pd.read_csv(path("TS3_Raw_tree_data.csv"))
ds_group = ds.groupby("TreeType")
ds_group = ds_group.get_group("BDL")
ds_group = ds_group.groupby("Age")[["DBH (cm)", "TreeHt (m)", "Leaf (m2)"]]
ds_group = ds_group.mean()

DBH  = ds_group["DBH (cm)"]/100
H    = ds_group["TreeHt (m)"]/1
leaf = ds_group["Leaf (m2)"]

import matplotlib.pyplot as plt
import numpy as np

R       = DBH/2
density = 800
v       = density* H*2*np.pi*(R)**2 

data = pd.DataFrame({"Age":v.index[1:], "H":H.values[1:], "DBH":DBH.values[1:], "val":v.values[1:]})
data.drop(data[ ((data["Age"] > 100) & (data["Age"] > 0) & (data["val"] < 1.6e4)) | ((data["Age"] > 176) & (data["val"] < 9e4)) ].index, inplace=True)

V_max = max(data["val"])
Vscaled = data["val"]/V_max

# ----------------------------------------------------------------------- Poly
t_poly = np.linspace(data["Age"].min() , data["Age"].max(), 1000)
degs = [1,2,3]
X = np.column_stack([data["Age"]**n for n in degs])
coeffs = np.linalg.pinv(X.T@X)@X.T@ data["val"].values
y_poly = np.sum([coeff*t_poly**n for n,coeff in zip(degs,coeffs)], axis=0)/V_max

# ----------------------------------------------------------------------- Sigmoid
from scipy.optimize import curve_fit
def fsigmoid(x, a, b, c,d):
    return d/(1.0 + np.exp(-a*(x-b)))+c

def fsigmoid(x, a, b, c):
    return 1/(1.0 + np.exp(-a*(x-b)))+c

popt, pcov = curve_fit(fsigmoid, data["Age"].values, Vscaled, method='dogbox', p0=[0.02,np.median(X),0])#bounds=([0.,0.,-1,1],[0.1, 300.,1, 10]))
y_sig = fsigmoid(data["Age"].values,*popt)

# ----------------------------------------------------------------------- Chapman-Richards
t_cr = data["Age"].values
def chapman_richards(x, a, b, m):
    return a*(1-np.exp(-b*x))**m

popt, pcov = curve_fit(chapman_richards, data["Age"].values, Vscaled, method='dogbox', bounds=([0.3,0.02,1],[1.5, 0.2, 15]))
y_cr = chapman_richards(data["Age"].values,*popt)

# ----------------------------------------------------------------------- Standard alometric equation
t_stdal = data["Age"].values

def std_alo(x, a, b):
    return a*x**b

popt, pcov = curve_fit(std_alo, data["Age"].values, Vscaled, method='dogbox', p0=[10,3])#bounds=([0.,0.,-1,1],[0.1, 300.,1, 10]))
y_std_alo = std_alo(data["Age"],*popt)

# ----------------------------------------------------------------------- Plot Volume vs Time
fig,axs = plt.subplots(1)

axs.plot(t_cr, y_sig,color="green", label="Sigmoid eq.")
axs.plot(t_poly, y_poly,color="blue", label="Polynomial eq.")
axs.plot(t_cr, y_cr, color="red", label="Chapman-Richard eq.")
axs.plot(t_stdal,y_std_alo, color="black",label="Standard Alometric eq.")
axs.set_ylabel("Weight in Kg")
axs.set_xlabel("Age")
axs.scatter(data["Age"],Vscaled, color="grey")
axs.legend()
axs.grid()

# ----------------------------------------------------------------------- DBH vs height
fig,axs = plt.subplots(1)
axs.scatter(DBH,H)
axs.grid()
axs.set_xlabel("Height")
axs.set_ylabel("DBH(H)")
# ----------------------------------------------------------------------- DBH vs time
fig,axs=plt.subplots(1)
axs.scatter(data["Age"], data["DBH"])
axs.grid()
axs.set_xlabel("Age")
axs.set_ylabel("DBH(t)")

# ----------------------------------------------------------------------- height vs time
fig,axs=plt.subplots(1)
axs.scatter(data["Age"], data["H"])
axs.grid()
axs.set_xlabel("Age")
axs.set_ylabel("H(t)")

# ----------------------------------------------------------------------- Perturbation 
from scipy.interpolate import CubicSpline
np.random.seed(1692958112)

# Example of a random spline perturbation function
n_ctrl_pts = 5
x_ctrl = np.linspace(t_poly[0], t_poly[-1], n_ctrl_pts)
y_ctrl = np.random.random(size=n_ctrl_pts) # range -> [0,1)
y_ctrl[0] = y_ctrl[-1] = 0 # fix boundary to avoid a shift, i.e change in the function
spline = CubicSpline(x_ctrl, y_ctrl*10)
y_poly_pert = (y_poly + spline(t_poly)/60) + np.random.normal(size=len(y_poly))/25

fig,axs=plt.subplots(1)
axs.plot(y_poly, linestyle="--", label="Polynomial function")
axs.plot(y_poly_pert, label="Perturbated polynomial function")
axs.grid()
axs.legend()

plt.show()
