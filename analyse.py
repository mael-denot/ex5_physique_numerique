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


#evolution analysis of mean velocity for different value of N_part
def mean_velocity_evolution(params):
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
    plt.title("evolution of mean velocity")
    plt.xlabel("N_part")
    plt.ylabel("Error")
    plt.hlines(0.0, 0, 3500, linestyles='dashed', colors='black')
    plt.legend()
    plt.show()


# mean_velocity_evolution(parameters)

#evolution analysis of variance of velocity for different value of N_part
def variance_velocity_evolution(params):


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
    plt.title("evolution of variance of velocity")
    plt.xlabel("N_part")
    plt.ylabel("Error")
    plt.hlines(0.0, 0, 3500, linestyles='dashed', colors='black')
    plt.legend()
    plt.show()

#variance_velocity_evolution(parameters)

#simultaneous evolution analysis of mean velocity and variance of velocity for different value of N_partwith different scales
def simultaneous_evolution(params):
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
    #ax1.plot(N_parts, (4.1/100.)*np.exp(-0.01*N_parts+0.003), color='black', linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('variance of velocity', color=color)  # we already handled the x-label with ax1
    ax2.plot(N_parts, errors2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # add exponential fit to the data
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()



#simultaneous_evolution(parameters)

# c)

def analytical_mean(t):
    return parameters['vc']+(parameters['v0']-parameters['vc'])*np.exp(-parameters['gamma']*t)

def analytical_variance(t, D=parameters['D']):
    print(t, D)
    return parameters['D']/parameters['gamma']+(parameters['sigma0']**2-parameters['D']/parameters['gamma'])*np.exp(-2*parameters['gamma']*t)

#update parameters
parameters['Initial_distrib'] = 'G'
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
    #plot analytical solution
    plt.plot(t, analytical_mean(t), linestyle='dotted', color='red',label='Analytical')
    plt.title("Mean velocity")
    plt.xlabel("time")
    plt.ylabel("mean velocity")
    plt.legend()
    plt.show()
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(t, v_variance, label='Simulated')
    #plot analytical solution
    plt.plot(t, analytical_variance(t), linestyle='dotted', color='red', label='Analytical')
    plt.title("Variance of velocity")
    plt.xlabel("time")
    plt.ylabel("variance of velocity")
    plt.legend()
    plt.show()

#mean_velocity_variance_velocity(parameters)

# run simulations for five values of N_part and plot mean velocity and variance of velocity at each time step and put them on the same plot
def mean_velocity_variance_velocity_Npart(params):
    #initialise a 5x2000 matrix to store the data
    v_mean = np.zeros((5, 1001))
    var_mean = np.zeros((5, 1001))
    #do a loop over five different values of N_part and store the data in the matrix
    for i, N_part in enumerate([20, 50, 100, 200, 500]):
        params['N_part'] = N_part
        simulationData = runSimulation("adMC", params)
        v_mean[i,:] = simulationData[:,2]
        var_mean[i,:] = simulationData[:,3]
    
    #plot the data
    t= simulationData[:,0]
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(t, v_mean[0,:], label='N_part = 20')
    plt.plot(t, v_mean[1,:], label='N_part = 50')
    plt.plot(t, v_mean[2,:], label='N_part = 100')
    plt.plot(t, v_mean[3,:], label='N_part = 200')
    plt.plot(t, v_mean[4,:], label='N_part = 500')
    plt.title("Mean velocity")
    plt.xlabel("time")
    plt.ylabel("mean velocity")
    plt.legend()
    plt.show()

    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(t, var_mean[0,:], label='N_part = 20')
    plt.plot(t, var_mean[1,:], label='N_part = 50')
    plt.plot(t, var_mean[2,:], label='N_part = 100')
    plt.plot(t, var_mean[3,:], label='N_part = 200')
    plt.plot(t, var_mean[4,:], label='N_part = 500')
    plt.title("Variance of velocity")
    plt.xlabel("time")
    plt.ylabel("variance of velocity")
    plt.legend()
    plt.show()



#mean_velocity_variance_velocity_Npart(parameters)

#evolution study of mean velocity at t=tfin for increasing nsteps values
def mean_velocity_evolution(params):
    params['Initial_distrib'] = 'G'
    params['v0'] = 0.0
    params['sigma0'] = 0.1
    params['tfin'] = 60.0

    #nsteps is a logarithmic scale going from 10 to 10000
    nsteps = np.logspace(1, 4, 10)
    errors = np.zeros((len(nsteps)))

    #take the last element of the third column of the simulation data which is the mean velocity at tfin
    for i, nstep in enumerate(nsteps):
        params['nsteps'] = int(nstep)
        simulationData = runSimulation("adMC", params)
        errors[i] = np.absolute(simulationData[-1,2] - 0.0)

    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(nsteps, errors, label='Simulated')
    plt.title("evolution of mean velocity")
    plt.xlabel("nsteps")
    plt.ylabel("mean velocity")
    plt.legend()
    plt.show()

#mean_velocity_evolution(parameters)

# run mean_velocity_variance_velocity_Npart for different values of vc and sigma0

# parameters['v0'] = 3.0
# parameters['sigma0'] = 10.0
#mean_velocity_variance_velocity(parameters)

#evolution study of mean velocity at t=tfin for values of D from 0.01 to 1.0
def mean_velocity_evolution_D(params):
    params['Initial_distrib'] = 'G'
    params['v0'] = 0.0
    params['sigma0'] = 0.1
    params['tfin'] = 60.0
    params['vc'] = 0.0

    #D is a logarithmic scale going from 0.01 to 1.0
    D = np.logspace(-2, 0, 10)
    errors = np.zeros((len(D)))

    #take the last element of the third column of the simulation data which is the mean velocity at tfin
    for i, d in enumerate(D):
        params['D'] = d
        params['gamma'] = params['D']/2.5
        simulationData = runSimulation("adMC", params)
        errors[i] = np.absolute(simulationData[-1,2] - 0.0)

    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(D, errors, label='Simulated')
    plt.title("Evolution of mean velocity")
    plt.xlabel("D")
    plt.ylabel("mean velocity")
    plt.legend()
    plt.show()

#study of variance of velocity at tfin for values of D from 0.01 to 1.0
def variance_velocity_evolution_D(params):
    params['Initial_distrib'] = 'G'
    params['v0'] = 0.0
    params['sigma0'] = 0.1
    params['tfin'] = 60.0
    params['vc'] = 0.0

    #D is a logarithmic scale going from 0.01 to 1.0
    D = np.logspace(-2, 0, 20)
    errors = np.zeros((len(D)))
    theoretical = np.zeros((len(D)))

    #take the last element of the third column of the simulation data which is the mean velocity at tfin
    for i, d in enumerate(D):
        params['D'] = d
        #params['gamma'] = params['D']/2.5
        simulationData = runSimulation("adMC", params)
        errors[i] = np.absolute(simulationData[-1,3] - 0.0)
        theoretical[i] = analytical_variance(params['tfin'], d)

    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(D, errors, label='Simulated')
    plt.plot(D, theoretical, label='Theoretical')
    plt.title("Evolution of variance of velocity")
    plt.xlabel("D")
    plt.ylabel("variance of velocity")
    plt.legend()
    plt.show() 

#variance_velocity_evolution_D(parameters)

#plot the speed in a histogram
def speed_histogram(params, rang=-1):
    simulationData = runSimulation("adMC", params)
    speed = simulationData[rang,4:]
    
    a = np.linspace(params['vlb'], params['vhb'], params['N_bins'])

    #plot the histogram
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.hist(a, params['N_bins'], weights=speed)
    plt.title("Speed histogram")
    plt.xlabel("speed")
    plt.ylabel("frequency")
    plt.show()

#parameters['initial_distrib'] = 'D'
parameters['initial_distrib'] = input("initial distribution (D, G, U, C):")
parameters['v0'] = 0.0
parameters['sigma0'] = 3.0
parameters['vd_D'] = 4.0
parameters['vg_D'] = -4.0
parameters['tfin'] = 1.0
parameters['sampling'] = 2
parameters['nsteps'] = 100
speed_histogram(parameters, 0)

#plot the particles in the 14th bin
def particles_in_bin(params, bin):
    simulationData = runSimulation("adMC", params)
    particles = simulationData[:,4+bin]
    t = simulationData[:,0]
    
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(t, particles)
    plt.title("Particles in bin")
    plt.xlabel("time")
    plt.ylabel("number of particles")
    plt.show()

# particles_in_bin(parameters, 14)

print ("code finished")