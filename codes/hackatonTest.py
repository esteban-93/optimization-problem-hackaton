import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
from pyomo.opt import SolverStatus,TerminationCondition
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

ROWS_TO_TRAIN = 100000
FACTOR = 0.001
INCHES_TO_FEETS = 12
WIDTHMAX = 96/12
DEPTHMAX = 96/12
HEIGHTMAX = 96/12
M = 1/5
PENALTY_FACTOR = 10

# Put the absolute path where the file is
path = "/home/helmholtz/Desktop"
file = "report1614289776153_random.xlsx"
data = pd.read_excel(os.path.join(path, file))

data_train = data.iloc[:ROWS_TO_TRAIN, :].copy()
data_train[["Width", "Height", "Length"]] = data_train[["Width", "Height", "Length"]]/INCHES_TO_FEETS

data_train_to_model = data_train[["Weight", "Cubic Feet"]].copy()
data_train_to_model["Weight"] = data_train_to_model["Weight"]*FACTOR
data_train_to_model["ind"] = 1
data_train_to_model["m^2"] = data_train_to_model["Weight"]**2
data_train_to_model["m^3"] = data_train_to_model["Weight"]**3
data_train_to_model["v^2"] = data_train_to_model["Cubic Feet"]**2
data_train_to_model["v^3"] = data_train_to_model["Cubic Feet"]**3
data_train_to_model["log(v)"] = data_train_to_model[["Cubic Feet"]].apply(lambda x: np.log(x))
data_train_to_model["m*v"] = data_train_to_model[["Weight", "Cubic Feet"]].\
    apply(lambda x: x[0]*x[1], axis=1)
data_train_to_model["(m*v)^2"] = data_train_to_model["m*v"]**2

m = pyo.ConcreteModel()

# set
m.samples = pyo.Set(initialize=list(range(data_train_to_model.shape[0])),
    doc="number of instances")
m.features = pyo.Set(initialize=data_train_to_model.columns.values.tolist(), 
    doc="number of features")

# Params
m.lambd = pyo.Param(initialize=M, doc="punish error factor")

# Variables
m.subErrW = pyo.Var(m.samples, domain=pyo.PositiveReals, 
                    doc="subestimation error of width")
m.oveErrW = pyo.Var(m.samples, domain=pyo.PositiveReals, 
                    doc="overestimation error of width")
m.subErrH = pyo.Var(m.samples, domain=pyo.PositiveReals,
                    doc="subestimation error of height")
m.oveErrH = pyo.Var(m.samples, domain=pyo.PositiveReals, 
                    doc="overestimation error of height")
m.subErrD = pyo.Var(m.samples, domain=pyo.PositiveReals,
                    doc="subestimation error of depth")
m.oveErrD = pyo.Var(m.samples, domain=pyo.PositiveReals, 
                    doc="overestimation error of depth")
m.subErrN = pyo.Var(m.samples, domain=pyo.PositiveReals,
                    doc="subestimation error of handling units")
m.oveErrN = pyo.Var(m.samples, domain=pyo.PositiveReals, 
                    doc="overestimation error of handling units")
m.w = pyo.Var(m.samples, bounds=(np.NINF, np.log(WIDTHMAX)), doc="natural logarithm of width")
m.h = pyo.Var(m.samples, bounds=(np.NINF, np.log(HEIGHTMAX)), doc="natural logarithm of height")
m.d = pyo.Var(m.samples, bounds=(np.NINF, np.log(DEPTHMAX)), doc="natural logarithm of depth")
m.n = pyo.Var(m.samples, bounds=(0, np.infty), doc="natural logarithm of handling units")
m.betaW = pyo.Var(m.features, doc="regression coefficients of width function")
m.betaH = pyo.Var(m.features, doc="regression coefficients of height function")
m.betaD = pyo.Var(m.features, doc="regression coefficients of depth function")
m.betaN = pyo.Var(m.features, doc="regression coefficients of handling units function")

# Objective function
def obj(m):  
    return 1/(4*len(m.samples))*sum(m.lambd*(m.subErrW[i] + m.subErrH[i] + 
               m.subErrD[i] + PENALTY_FACTOR*m.subErrN[i]) +
               (1-m.lambd)*(m.oveErrW[i] + m.oveErrH[i] + m.oveErrD[i] + 
               PENALTY_FACTOR*m.oveErrN[i]) for i in m.samples)
m.mape = pyo.Objective(rule=obj, sense=pyo.minimize, doc="Error minimization process")

Width = data_train["Width"].to_dict()
Height = data_train["Height"].to_dict()
Depth = data_train["Length"].to_dict()
HU = data_train["HU Count"].to_dict()

# Constraints
##### Widths
def constraint_1(m, i):
    return m.subErrW[i] >= np.log(Width[i]) - m.w[i]
m.subestimErrorW = pyo.Constraint(m.samples, rule=constraint_1, doc="")

def constraint_2(m, i):
    return m.oveErrW[i] >= m.w[i] - np.log(Width[i])
m.overestimErrorW = pyo.Constraint(m.samples, rule=constraint_2, doc="")

def constraint_3(m, i):
    return m.w[i] == sum(m.betaW[j]*data_train_to_model.loc[i,j] for j in m.features)
m.regW = pyo.Constraint(m.samples, rule=constraint_3, doc="")

##### Heights
def constraint_4(m, i):
    return m.subErrH[i] >= np.log(Height[i]) - m.h[i]
m.subestimErrorH = pyo.Constraint(m.samples, rule=constraint_4, doc="")

def constraint_5(m, i):
    return m.oveErrH[i] >= m.h[i] - np.log(Height[i])
m.overestimErrorH = pyo.Constraint(m.samples, rule=constraint_5, doc="")

def constraint_6(m, i):
    return m.h[i] == sum(m.betaH[j]*data_train_to_model.loc[i,j] for j in m.features)
m.regH = pyo.Constraint(m.samples, rule=constraint_6, doc="")

##### Depth
def constraint_7(m, i):
    return m.subErrD[i] >= np.log(Depth[i]) - m.d[i]
m.subestimErrorD = pyo.Constraint(m.samples, rule=constraint_7, doc="")

def constraint_8(m, i):
    return m.oveErrD[i] >= m.d[i] - np.log(Depth[i])
m.overestimErrorD = pyo.Constraint(m.samples, rule=constraint_8, doc="")

def constraint_9(m, i):
    return m.d[i] == sum(m.betaD[j]*data_train_to_model.loc[i,j] for j in m.features)
m.regD = pyo.Constraint(m.samples, rule=constraint_9, doc="")

##### Handling units
def constraint_10(m, i):
    return m.subErrN[i] >= np.log(HU[i]) - m.n[i]
m.subestimErrorN = pyo.Constraint(m.samples, rule=constraint_10, doc="")

def constraint_11(m, i):
    return m.oveErrN[i] >= m.n[i] - np.log(HU[i])
m.overestimErrorN = pyo.Constraint(m.samples, rule=constraint_11, doc="")

def constraint_12(m, i):
    return m.n[i] == sum(m.betaN[j]*data_train_to_model.loc[i,j] for j in m.features)
m.regN = pyo.Constraint(m.samples, rule=constraint_12, doc="")


e = {}
for i in data_train_to_model.columns.values.tolist():
    if i == 'log(v)':
        e[i] = 1
    else:
        e[i] = 0

def constraint_13(m, j):
    return (m.betaN[j] + m.betaD[j] + m.betaW[j] + m.betaH[j]) == e[j]
m.vol = pyo.Constraint(m.features, rule=constraint_13)


opt = pyo.SolverFactory("ipopt")
results = opt.solve(m)

# Check status
if (results.solver.status == SolverStatus.ok) and \
        (results.solver.termination_condition == TerminationCondition.optimal):
    print("Optimal solution ")
    print(str(results.solver)) 
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("Something is wrong..")
else:
    # something is wrong
    print(str(results.solver))

m.betaW.display()
m.betaH.display()
m.betaD.display()
m.betaN.display()

########################
#   Processing data
########################
df = pd.DataFrame()
df["betaH"] = [m.betaH.get_values()[key] for key in m.betaH.get_values()]
df["betaN"] = [m.betaN.get_values()[key] for key in m.betaN.get_values()]
df["betaD"] = [m.betaD.get_values()[key] for key in m.betaD.get_values()]
df["betaW"] = [m.betaW.get_values()[key] for key in m.betaW.get_values()]
df.to_csv("betasUpdated.csv", index=False)

widths = {key: np.exp(m.w.get_values()[key]) for key in m.w.get_values()}
N = {key: np.exp(m.n.get_values()[key]) for key in m.n.get_values()}
depths = {key: np.exp(m.d.get_values()[key]) for key in m.d.get_values()}
heights = {key: np.exp(m.h.get_values()[key]) for key in m.h.get_values()}

widths = list(widths.values())
N = list(N.values())
depths = list(depths.values())
heights = list(heights.values())

# Calculating volumen
volumen = np.array(widths)*np.array(depths)*np.array(heights)*np.array(N)
volumen = list(volumen)

dimensions_train = pd.DataFrame()
dimensions_train["width-train"] = widths
dimensions_train["length-train"] = depths
dimensions_train["heights-train"] = heights
dimensions_train["N-train"] = N
dimensions_train = pd.concat([dimensions_train, 
    data_train[["Height", "Width", "Length", "HU Count"]]], axis=1)

dimensions_train["mapeH-train"] = dimensions_train[["heights-train", "Height"]].\
    apply(lambda x: abs(x[0]/x[1] - 1)*100/dimensions_train.shape[0], axis=1)
dimensions_train["mapeW-train"] = dimensions_train[["width-train", "Width"]].\
    apply(lambda x: abs(x[0]/x[1] - 1)*100/dimensions_train.shape[0], axis=1)
dimensions_train["mapeL-train"] = dimensions_train[["length-train", "Length"]].\
    apply(lambda x: abs(x[0]/x[1] - 1)*100/dimensions_train.shape[0], axis=1)
dimensions_train["mapeN-train"] = dimensions_train[["N-train", "HU Count"]].\
    apply(lambda x: abs(x[0]/x[1] - 1)*100/dimensions_train.shape[0], axis=1)

mapeW_train = dimensions_train["mapeW-train"].sum()
mapeH_train = dimensions_train["mapeH-train"].sum()
mapeD_train = dimensions_train["mapeL-train"].sum()
mapeN_train = dimensions_train["mapeN-train"].sum()

print(f"EL mape de width es:{mapeW_train}%")
print(f"EL mape de height es:{mapeH_train}%")
print(f"EL mape de depth es:{mapeD_train}%")
print(f"EL mape de N es:{mapeN_train}%")

print("widths:\n", widths[0:30])
print("N's:\n", N[0:30])
print("depths:\n", depths[0:30])
print("heights:\n", heights[0:30])
print("volumenes:\n", volumen[0:30])


