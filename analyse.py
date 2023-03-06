import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Function to run c++ file for a set of parameters
def runSimulation(cppFile, params):
    '''
    Runs c++ simulation on input dictionary params and returns results as a matrix.
    Creates and then deletes two temporary files to pass parameters into and output data out of the c++ code.
    Results matrix contains output at each timestep as each row. Each variable is it's own column.

    Inputs:
        cppFile: String,    Name of c++ file
        params: dict,   Simulation Parameters

    Outputs:
        simulationData: np.array(nsteps+1, k),  matrix containing all output data across k variables
    '''

    # Set name of temporary output data file
    tempDataFileName = 'output.out'
    params['output'] = tempDataFileName

    # Create temporary file of parameters to get passed into simulation. Deleted after use.
    tempParamsFileName = 'configuration_temp.in'
    with open(tempParamsFileName, 'w') as temp:
        for key in params.keys():
            temp.write("%s %s %s\n" % (key, "=", params[key]))

    # Run simulation from terminal, passing in parameters via temporary file
    os.system(' .\\' + cppFile + ".exe .\\" + tempParamsFileName)

    # Save output data in simData matrix
    simulationData = np.loadtxt(tempDataFileName)

    # Delete temp files
    # os.remove(tempParamsFileName)
    # os.remove(tempDataFileName)

    return simulationData


parameters = {
'tfin' : 100.0,
'nsteps' : 2000,
'output' : 'output.out',
'sampling' : 2,
'N_part' : 1000,
'N_bins' : 30,
'v0' : -3.5,
'gamma' : 0.2,
'D' : 0.3, 
'vhb' : 5,
'vlb' : -5,
'vg_D' : -4,
'vd_D' : 4,
'sigma0':0.04,
'vc' : 3, 
'initial_distrib' : 'Gaussian',
}

# question b) initial conditions

def initial_conditions(params):
    params['Initial_distrib'] = 'G'
    params['v0'] = 0.0
    params['sigma0'] = 0.1
    params['tfin'] = 0.0
    # Run simulation
    simulationData = runSimulation("adMC", params)

    return simulationData[2], simulationData[3]

#print (initial_conditions(parameters))


#convergence analysis of mean velocity for different value of N_part
def mean_velocity_convergence(params):
    N_parts = np.logspace(1, 3.5, 50, base=10.0)
    errors = np.zeros((len(N_parts)))
    for i, N_part in enumerate(N_parts):
        params['N_part'] = N_part
        simulationData = runSimulation("adMC", params)
        v_mean = simulationData[2]
        errors[i] = np.absolute(v_mean - 0.0)
    plt.figure()
    #plt.ion()
    plt.rcParams.update({'font.size': 15})
    plt.plot(N_parts, errors, label='Simulated')
    plt.title("Convergence of mean velocity")
    plt.xlabel("N_part")
    plt.ylabel("Error")
    plt.hlines(0.0, 0, 3500, linestyles='dashed', colors='black')
    plt.legend()
    plt.show()


# mean_velocity_convergence(parameters)

#convergence analysis of variance of velocity for different value of N_part
def variance_velocity_convergence(params):


    N_parts = np.logspace(1, 3, 50, base=10.0)
    errors1 = np.zeros((len(N_parts)))
    errors2 = np.zeros((len(N_parts)))
    errors3 = np.zeros((len(N_parts)))

    for i, N_part in enumerate(N_parts):
        params['N_part'] = N_part
        simulationData = runSimulation("adMC", params)
        v_variance = simulationData[3]
        errors1[i] = np.absolute(v_variance - 0.01)

    for i, N_part in enumerate(N_parts):
        params['N_part'] = N_part
        simulationData = runSimulation("adMC", params)
        v_variance = simulationData[3]
        errors2[i] = np.absolute(v_variance - 0.01)
    
    for i, N_part in enumerate(N_parts):
        params['N_part'] = N_part
        simulationData = runSimulation("adMC", params)
        v_variance = simulationData[3]
        errors3[i] = np.absolute(v_variance - 0.01)

    plt.figure()
    #plt.ion()
    plt.rcParams.update({'font.size': 15})
    #plotting the errors with different colors
    plt.plot(N_parts, errors1, linewidth=2, color='red', label='Simulated')
    plt.plot(N_parts, errors2,  linewidth=1, color='blue', label='Simulated')
    #plt.plot(N_parts, errors3, color='green', label='Simulated')
    plt.title("Convergence of variance of velocity")
    plt.xlabel("N_part")
    plt.ylabel("Error")
    plt.hlines(0.0, 0, 3500, linestyles='dashed', colors='black')
    plt.legend()
    plt.show()

#variance_velocity_convergence(parameters)

#simultaneous convergence analysis of mean velocity and variance of velocity for different value of N_partwith different scales
def simultaneous_convergence(params):
    params['Initial_distrib'] = 'G'
    params['v0'] = 0.0
    params['sigma0'] = 0.1
    params['tfin'] = 0.0

    N_parts = np.logspace(1, 3, 50, base=10.0)
    errors1 = np.zeros((len(N_parts)))
    errors2 = np.zeros((len(N_parts)))

    for i, N_part in enumerate(N_parts):
        params['N_part'] = N_part
        simulationData = runSimulation("adMC", params)
        v_mean = simulationData[2]
        v_variance = simulationData[3]
        errors1[i] = np.absolute(v_mean - 0.0)
        errors2[i] = np.absolute(v_variance - 0.01)

    fig, ax1 = plt.subplots()
    #plt.rcParams.update(text.usetex : True )
    color = 'tab:red'
    ax1.set_xlabel('N_part')
    ax1.set_ylabel('mean velocity', color=color)
    ax1.plot(N_parts, errors1, color=color)
    ax1.plot(N_parts, (4.1/100.)*np.exp(-0.01*N_parts+0.003), color='black', linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('variance of velocity', color=color)  # we already handled the x-label with ax1
    ax2.plot(N_parts, errors2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # add exponential fit to the data
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()



#simultaneous_convergence(parameters)

# c)

#update parameters
parameters['vc'] = 2
parameters['D'] = 0.3
parameters['gamma'] = parameters['D']/2.5
parameters['v0'] = -2.5
parameters['sigma0'] = 0.1
parameters['tfin'] = 60.0

#run simulation and plot mean velocity and variance of velocity at each time step
def mean_velocity_variance_velocity(params):
    simulationData = runSimulation("adMC", params)
    v_mean = simulationData[:,2]
    v_variance = simulationData[:,3]
    t= simulationData[:,0]
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(t, v_mean, label='Simulated')
    plt.title("Mean velocity")
    plt.xlabel("time")
    plt.ylabel("mean velocity")
    plt.legend()
    plt.show()
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(t, v_variance, label='Simulated')
    plt.title("Variance of velocity")
    plt.xlabel("time")
    plt.ylabel("variance of velocity")
    plt.legend()
    plt.show()

mean_velocity_variance_velocity(parameters)



print ("code finished")
