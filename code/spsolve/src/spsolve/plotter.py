import math
import time 

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import root as scipy_root
from scipy.optimize import minimize as scipy_minimize

import solver

# Physical constants
h_bar = 0.276042828 # eV s
m_eff = 1.08 # effective mass of electron
PERMETTIVITY = 0.055263494 # q_e/(V*nm) 
q_e = 1 # elementary charge

def plot_distributions(grid, n_e, band, psi, energies, title=None, n_e_fit = None, L = 20):
    fig_pois, ax_pois = plt.subplots(3)
    
    # CHARGE DISTRIBUTION #
    ax_pois[0].plot(grid, n_e, label='Numerical')
    ax_pois[0].set_title('Charge distribution')
    ax_pois[0].set_xlabel('Position (nm)')
    #ax_pois[0].set_ylim([-1, 1])
    ax_pois[0].set_ylabel(r'Charge ($q_e/nm^3$)')
    if not isinstance(n_e_fit, type(None)):
        ax_pois[0].plot(grid[1:-1], n_e_fit[1:-1], label='Analytical')
        ax_pois[0].legend()
    
    # POTENTIAL # 
    ax_pois[1].plot(grid, band)
    #ax_pois[1].set_ylim([-1, 1])
    ax_pois[1].set_title('Conduction band')
    ax_pois[1].set_xlabel('Position (nm)')
    ax_pois[1].set_ylabel('Energy (eV)')
    
    # WAVEFUNCTION#
    #print('Normalize numerical: ', np.sum(psi[:,0]**2*grid[1]))
    ax_pois[2].plot(grid, psi[:,0], label='$E_0$ = {:f} eV'.format(energies[0])) 
                    
    if not isinstance(n_e_fit, type(None)):
        fit = math.sqrt(2/L)*np.sin(math.pi/L*grid)
        print('Normalize fit: ', np.sum(fit**2)*grid[1])
        ax_pois[2].plot(grid[1:-1], fit[1:-1], label='Analytical $E_1$ = {:f} eV'.format(math.pi**2*h_bar**2/(2*m_eff*20*2)+.799))
    else:
        number_of_wavefunctions = 3
        for n in np.arange(number_of_wavefunctions-1)+1:
            ax_pois[2].plot(grid, psi[:,n], label='$E_{}$ = {:f} eV'.format(n, energies[n]))
            
    
    ax_pois[2].legend(fontsize='xx-small')
    ax_pois[2].set_title('Wavefunction')
    ax_pois[2].set_xlabel('Position (nm)')
    ax_pois[2].set_ylabel('$\psi$')

    fig_pois.tight_layout()
    if type(title) not None:
        plt.savefig(title)
    plt.show()
    
def plot_conduction_band(grid, band, title='figures/conduction_band.png'):
    plt.plot(grid, band)
    plt.xlabel('Distance (nm)')
    plt.ylabel('Energy (eV)')
    plt.savefig(title)

def plot_mult_iteration(number_of_iterations = 1, fit = False):
    # System
    L = 20 # length of the system (nm)
    N = 500 # number of gridpoints
    T = 0 # Kelvin
    m_eff = 1.08
    grid = np.linspace(0, L, N)
    dl = grid[1]
    E_fermi = 0.8 # eV

    # Boundary conditions
    dirichlet_left = True
    potential_left_boundary = 0
    dirichlet_right = True
    potential_right_boundary = -5

    n_e = np.zeros(N)
    potential = np.zeros(N)
    psi = np.zeros(N)

    # Poisson matrix
    pois_matrix = solver.comp_pois_matrix(dl, N, dirichlet_left, dirichlet_right)
    syst = solver.make_system(dl, N, m_eff)
    
    for i in np.arange(number_of_iterations):
        # Solve for potential with Poisson equation
        phi = solver.solve_poisson(pois_matrix, n_e, dl, potential_left_boundary, potential_right_boundary)

        # Solve for wavefunction with Schrodinger equation
        energies, psi = solver.solve_syst(syst, phi, dl)
        #energies, psi = solver.solve_schrod(phi, dl)
        
        n_e_old = n_e.copy()
        n_e = solver.solve_charge_dist(psi, energies, E_fermi, T)
    
    if fit == True:
        # FIT #
        E_n = math.pi**2*h_bar**2/(2*m_eff*L**2)-potential_left_boundary
        print('Energy: ', E_n)
        n_e_fit = -q_e * 2*m_eff/(L*math.pi*h_bar**2) * np.sin(math.pi/L*grid)**2 * (E_fermi - E_n)
        title = 'iterations_' + str(number_of_iterations) + '_fit.png'
        plot_distributions(grid, n_e, phi, psi, energies, title=title, n_e_fit = n_e_fit, L = L)
    else:
        # PLOT #
        title = 'iterations_' + str(number_of_iterations) + '.png'
        plot_distributions(grid, n_e, phi, psi, energies, title=title)
    
def plot_charge_density(startV = -1, stopV = 1, fit = False):
    # System
    l = 50 # length of the system (nm)
    N = 500 # number of gridpoints
    N_V = 50
    grid = np.linspace(0, l, N)
    dl = grid[1]
    E_fermi = 0.8 # eV

    # Boundary conditions
    dirichlet_left = True
    potential_left_boundary = 0
    dirichlet_right = False
    potential_right_boundary = 0
    
    potential = np.zeros(N)
    psi = np.zeros(N)
    
    # Poisson matrix
    pois_matrix = solver.comp_pois_matrix(dl, N, dirichlet_left, dirichlet_right)
    syst = solver.make_system(dl, N)

    n_e_mid = np.zeros(N_V)
    #-.7997, -.7999
    V = np.linspace(startV, stopV, N_V)
    
    for i in range(N_V):
        V_0 = V[i]
        n_e = np.zeros(N)
        # Solve for potential with Poisson equation
        phi = solver.solve_poisson(pois_matrix, n_e, dl, V_0, potential_right_boundary)

        # Solve for wavefunction with Schrodinger equation
        energies, psi = solver.solve_syst(syst, phi, dl)
        #energies, psi = solver.solve_schrod(phi, dl)
        
        n_e = solver.solve_charge_dist(psi, energies, E_fermi)
        n_e_mid[i] = n_e[int(N/2)]
        
    plt.plot(V, n_e_mid, label='One iter')
    plt.xlabel('$V_0$ (V)')
    plt.ylabel('$n_e$ ($q_e/nm^2$)')
    # Fit
    if fit == True:
        z = np.polyfit(V, n_e_mid, 1)
        y = V*z[0] + z[1]
        error = np.sum((y-n_e_mid)**2)/len(V)
        print('Error: ', error)
        plt.plot(V, y, label='fit')
        plt.legend()
        
    plt.savefig('different_V_0.png')
    
def plot_E_n():
    # System
    l = 20 # length of the system (nm)
    N = 500 # number of gridpoints
    N_V = 40
    grid = np.linspace(0, l, N)
    dl = grid[1]
    E_fermi = 0.8 # eV

    # Boundary conditions
    dirichlet_left = True
    potential_left_boundary = 0
    dirichlet_right = False
    potential_right_boundary = 0
    
    potential = np.zeros(N)
    psi = np.zeros(N)
    
    # Poisson matrix
    pois_matrix = solver.comp_pois_matrix(dl, N, dirichlet_left, dirichlet_right)
    syst = solver.make_system(dl, N)

    n_e_mid = np.zeros(N_V)
    V = np.linspace(-.799, -.801, N_V)
    E_n_max = 10
    E_n = np.zeros((E_n_max, N_V))
    
    for i in range(N_V):
        V_0 = V[i]
        n_e = np.zeros(N)
        # Solve for potential with Poisson equation
        phi = solver.solve_poisson(pois_matrix, n_e, dl, V_0, potential_right_boundary)

        # Solve for wavefunction with Schrodinger equation
        energies, psi = solver.solve_syst(syst, phi, dl)
        E_n[:, i] = energies[:E_n_max]
        
    for j in range(E_n_max):
        plt.plot(V, E_n[j,:], label='n = {}'.format(j))
        
    z = np.polyfit(V, E_n[0,:], 1)
    y = z[1] + z[0]*V
    error = np.sum((y-E_n[0,:])**2)/len(V)
    print('Fit error: ', error)
    
    plt.plot(V, y, label='Fit')
    plt.xlabel('$V_0$')
    plt.ylabel('$E_n$ (eV)')
    plt.legend()
    plt.savefig('E_n.png')
        
def plot_optimize():
    """
    Solve the self-consistency equation by optimizing the difference. 
    """
    # System
    l = 10 # length of the system (nm)
    N = 200 # number of gridpoints
    T = 0
    m_eff = 1.08
    grid = np.linspace(0, l, N)
    dl = grid[1]
    E_fermi = 0.8 # eV
    
    K = 0
    rho = np.ones(N)*K
    phi = np.zeros(N)
    energies = np.zeros(N)
    psi = np.zeros((N, N))
    errors = np.array([])
    
    # Boundary conditions
    dirichlet_left = True
    potential_left_boundary = 0
    dirichlet_right = True
    potential_right_boundary = 0
    
    # Poisson matrix
    pois_matrix = solver.comp_pois_matrix(dl, N, dirichlet_left, dirichlet_right)
    phi_init = solver.solve_poisson(pois_matrix, rho, dl, potential_left_boundary, potential_right_boundary)
    
    syst = solver.make_system(dl, N, m_eff)
    
    def comp_pot_distr(phi):
        # SOLVE SCHRODINGER EQUATION #
        # Solve for wavefunction with Schrodinger equation
        energies, psi = solver.solve_syst(syst, phi, dl)
    
        # COMPUTE NEW CHARGE DISTRIBUTION #
        rho = solver.solve_charge_dist(psi, energies, E_fermi, T)
        
        # SOLVE POISSON EQUATION #
        # Solve for potential with Poisson equation
        phi = solver.solve_poisson(pois_matrix, rho, dl, potential_left_boundary, potential_right_boundary)
        return phi
    
    def self_consistent(phi):
        phi_old = phi.copy()
        phi = comp_pot_distr(phi)
        
        # Compute error
        diff = np.real(np.squeeze(phi)-np.squeeze(phi_old))
        errors.resize(len(errors)+1)
        error = 1/2/rho.size*diff**2
        errors[-1] = np.sum(error)
        return diff
    
    optim_result = scipy_root(self_consistent, phi_init, tol = 0.00001)
    # options=dict(maxiter=1000)
    
    phi = optim_result.x
    
    #energies, psi = solver.solve_syst(syst, phi, dl)
    energies, psi = solver.solve_schrod(phi, dl)
    rho = solver.solve_charge_dist(psi, energies, E_fermi, T)
    #final_error = np.sum(1/2/n_e_final.size*(n_e_temp - n_e_final)**2)
    #print('Final error new: ', final_error)
    
    print('Final error: ', errors[-1])
    plt.plot(np.arange(errors.size), errors)
    plt.savefig('error.png')
    plt.show()
    plot_distributions(grid, rho, phi, psi, energies, title='optimize.png')
    
def plot_poisson(grid, phi, fit = False):
    """
    Solve the self-consistency equation by optimizing the difference. 
    """
    K = -.001
    n_e = np.ones(len(grid))*K # Initial charge distribution
    L = grid[-1]+grid[0]
    phi_analytical = -K/2/PERMETTIVITY/12.048 * grid**2 + K*L/2/PERMETTIVITY/12.048 * grid - 0.6
    
    # PLOT #
    fig_pois, ax_pois = plt.subplots(2)
    
    # CHARGE DISTRIBUTION #
    ax_pois[0].plot(grid, n_e, label='Numerical')
    ax_pois[0].set_title('Charge distribution')
    ax_pois[0].set_xlabel('Position (nm)')
    #ax_pois[0].set_ylim([-1, 1])
    ax_pois[0].set_ylabel(r'Charge ($q_e/nm^3$)')
    
    # POTENTIAL # 
    ax_pois[1].plot(grid, phi, label = 'Numerical')
    ax_pois[1].set_title('Potential distribution')
    ax_pois[1].set_xlabel('Position (nm)')
    ax_pois[1].set_ylabel('Band (eV)')
    #ax_pois[1].set_ylim([-1, 1])
    if fit == True:
        ax_pois[1].plot(grid, phi_analytical, label = 'Analytical')
        ax_pois[1].legend()
    
    fig_pois.tight_layout()
    plt.savefig('fit_poisson.png')
    plt.show()
    
'''
FUNCTIONS
plot_mult_iteration(number_of_iterations = 1, fit = False)
plot_charge_density(startV = -1, stopV = 1, fit = False)
plot_E_n()
plot_optimize()
plot_poisson()
'''
if __name__ == "__main__":
    plot_optimize()
    #plot_mult_iteration(number_of_iterations = 1, fit=False)
