import pandas as pd
import numpy as np
import pyomo.environ as pyo
import os
from pyomo.opt import SolverStatus,TerminationCondition
from sklearn.model_selection import train_test_split
from pyomo.util.infeasible import log_infeasible_constraints
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Put the absolute path here
path = "/home/helmholtz/Desktop/"
file = "x_test_set_1st_datathon.csv"
data = pd.read_csv(os.path.join(path, file))
betas = pd.read_csv(path + "betasUpdated.csv")

# test
# data_test = data.iloc[100001:, :].copy()
data_test = data.copy()
data_test["Cubic Feet"] = data_test["Cubic Feet"]
# data_test.reset_index(drop=True, inplace=True)
# data_test[["Width", "Height", "Length"]] = data_test[["Width", "Height", "Length"]]/12

data_test_model = data_test[["Weight", "Cubic Feet"]].copy()
data_test_model["Weight"] = data_test_model["Weight"]
data_test_model["ind"] = 1
data_test_model["m^2"] = data_test_model["Weight"]**2
data_test_model["m^3"] = data_test_model["Weight"]**3
data_test_model["v^2"] = data_test_model["Cubic Feet"]**2
data_test_model["v^3"] = data_test_model["Cubic Feet"]**3
data_test_model["log(v)"] = data_test_model[["Cubic Feet"]].apply(lambda x: np.log(x))
data_test_model["m*v"] = data_test_model[["Weight", "Cubic Feet"]].apply(lambda x: x[0]*x[1], axis=1)
data_test_model["(m*v)^2"] = data_test_model["m*v"]**2

data_test_model["Weight"].hist()

m = pyo.ConcreteModel()
opt = pyo.SolverFactory("ipopt")

WIDTHMAX = 96/12
DEPTHMAX = 96/12
HEIGHTMAX = 96/12
CONVERSION = WIDTHMAX*DEPTHMAX*HEIGHTMAX
M = 1/5

# set
m.features = pyo.Set(initialize=data_test_model.columns.values.tolist(), 
    doc="number of features")

# Params
m.lambd = pyo.Param(initialize=M)

# Variables
m.subErrW = pyo.Var(domain=pyo.PositiveReals, 
                    doc="subestimation error of width")
m.oveErrW = pyo.Var(domain=pyo.PositiveReals, 
                    doc="overestimation error of width")
m.subErrH = pyo.Var(domain=pyo.PositiveReals,
                    doc="subestimation error of height")
m.oveErrH = pyo.Var(domain=pyo.PositiveReals, 
                    doc="overestimation error of height")
m.subErrD = pyo.Var(domain=pyo.PositiveReals,
                    doc="subestimation error of depth")
m.oveErrD = pyo.Var(domain=pyo.PositiveReals, 
                    doc="overestimation error of depth")

m.w = pyo.Var(bounds=(np.NINF, np.log(WIDTHMAX)), doc="natural logarithm of width")
m.h = pyo.Var(bounds=(np.NINF, np.log(HEIGHTMAX)), doc="natural logarithm of height")
m.d = pyo.Var(bounds=(np.NINF, np.log(DEPTHMAX)), doc="natural logarithm of depth")

m.logheight = pyo.Param(default=0, mutable=True)
m.logwidth = pyo.Param(default=0, mutable=True)
m.logN = pyo.Param(default=0, mutable=True)
m.logdepth = pyo.Param(default=0, mutable=True)
m.volumen = pyo.Param(default=0, mutable=True)

# Objective function
def obj(m):  
    return (1/3)*(m.lambd*(m.subErrW + m.subErrH + m.subErrD) + \
        (1-m.lambd)*(m.oveErrW + m.oveErrH + m.oveErrD))
m.mape = pyo.Objective(rule=obj, sense=pyo.minimize, doc="Error minimization process")

# Constraints
##### Widths
def constraint_1(m):
    return m.subErrW >= m.logwidth - m.w
m.subestimErrorW = pyo.Constraint(rule=constraint_1, doc="")

def constraint_2(m):
    return m.oveErrW >= m.w - m.logwidth
m.overestimErrorW = pyo.Constraint(rule=constraint_2, doc="")

##### Heights
def constraint_4(m):
    return m.subErrH >= m.logheight - m.h
m.subestimErrorH = pyo.Constraint(rule=constraint_4, doc="")

def constraint_5(m):
    return m.oveErrH >= m.h - m.logheight
m.overestimErrorH = pyo.Constraint(rule=constraint_5, doc="")

##### Depth
def constraint_7(m):
    return m.subErrD >= m.logdepth - m.d
m.subestimErrorD = pyo.Constraint(rule=constraint_7, doc="")

def constraint_8(m):
    return m.oveErrD >= m.d - m.logdepth
m.overestimErrorD = pyo.Constraint(rule=constraint_8, doc="")

def constraint_13(m):
    return (m.d + m.w + m.h + m.logN) == m.volumen
m.vol = pyo.Constraint(rule=constraint_13)

width_for = []
height_for = []
length_for = []
N_for = []

for i in range(data_test_model.shape[0]):

    logheight = (np.array(betas["betaH"])*data_test_model.loc[i, :].to_numpy()).sum()
    logwidth = (np.array(betas["betaW"])*data_test_model.loc[i, :].to_numpy()).sum()
    logdepth = (np.array(betas["betaD"])*data_test_model.loc[i, :].to_numpy()).sum()
    logn_pre = (np.array(betas["betaN"])*data_test_model.loc[i, :].to_numpy()).sum()
    
    l1 = np.floor(np.exp(logn_pre))
    l2 = np.ceil(data_test.loc[i, "Cubic Feet"]/CONVERSION)
    n_cal = max(l1, l2)
    logN = np.log(n_cal)

    # Mutable parameters
    m.logheight = logheight
    m.logwidth = logwidth
    m.logN = logN
    m.logdepth = logdepth
    m.volumen = data_test_model.loc[i, "log(v)"]

    results = opt.solve(m)

    # Check status
    if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
        print("Optimal solution ")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("Something is wrong..")
    else:
        # something is wrong
        print(str(results.solver))

    # Real dimentions
    depth = np.exp(m.d.value)
    height = np.exp(m.h.value)
    width = np.exp(m.w.value)
   
    height_for.append(height)
    width_for.append(width)
    length_for.append(depth)
    N_for.append(n_cal)

    # Check if there is an infeasible cnstraint
    log_infeasible_constraints(m)

# cols = ["HU Count", "Length", "Width", "Height"]
# Processing results
# volumen = list(np.array(height_for)*np.array(width_for)*np.array(length_for)*np.array(N_for))

volumen = list(np.array(height_for)*np.array(width_for)*np.array(length_for)*np.array(N_for))

dimentions = pd.DataFrame()
dimentions["Height"] = height_for
dimentions["Width"] = width_for
dimentions["Length"] = length_for
dimentions["HU Count"] = N_for

dimentions["Height"] = dimentions["Height"]*12
dimentions["Width"] = dimentions["Width"]*12
dimentions["Length"] = dimentions["Length"]*12
dimentions["Volumen"] = volumen

# Saving the results
dimentions.to_csv("results_inchesVol.csv", index=False)
print(dimentions)

# dimentions = pd.concat([dimentions, data_test[["Height", "Width", "Length", "HU Count"]].\
#     reset_index(drop=True)], axis=1)

# dimentions["mapeH"] = dimentions[["Height-for", "Height"]].\
#     apply(lambda x: abs(x[0]/x[1] - 1)*100/dimentions.shape[0], axis=1)
# dimentions["mapeW"] = dimentions[["Width-for", "Width"]].\
#     apply(lambda x: abs(x[0]/x[1] - 1)*100/dimentions.shape[0], axis=1)
# dimentions["mapeD"] = dimentions[["Length-for", "Length"]].\
#     apply(lambda x: abs(x[0]/x[1] - 1)*100/dimentions.shape[0], axis=1)
# dimentions["mapeN"] = dimentions[["HU-for", "HU Count"]].\
#     apply(lambda x: abs(x[0]/x[1] - 1)*100/dimentions.shape[0], axis=1)

# print(dimentions)

# mapeW = dimentions["mapeW"].sum()
# mapeH = dimentions["mapeH"].sum()
# mapeD = dimentions["mapeD"].sum()
# mapeN = dimentions["mapeN"].sum()

# print(f"EL mape de width es: {mapeW}%")
# print(f"EL mape de height es: {mapeH}%")
# print(f"EL mape de depth es: {mapeD}%")
# print(f"EL mape de N es: {mapeN}%")


