# This requires pyaudi

try:
    from .nn_controller import Controller
except:
    print("Warning: could not import the Controller class. If  you need it make sure pyaudi is installed")

# This requires progressbar2
try:
    from .ode45 import rkf45, rkf45_gduals
except ImportError:
    print("Warning: could not import the ODE45 related functions. If  you need it make sure progressbar2 and numpy are installed")

# This requires the pyampl module. 
try:
    from .run_ampl import solve_model_6dof, solve_model_5dof
except ImportError:
    print("Warning: could not import the AMPL related functions. If  you need it make sure amplpy and numpy is installed")
    
# This requires the ekin module
try:
    from .ekin import *
except ImportError:
    print("Warning: could not import ekin modules. If you need it, make sure progressbar2, matplotlib, joblib, numpy and amplpy are installed")