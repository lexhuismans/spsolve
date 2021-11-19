import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from . import solver

# Physical constants
h_bar = 0.276042828  # eV s
m_eff = 1.08  # effective mass of electron
PERMETTIVITY = 0.055263494  # q_e/(V*nm)
q_e = 1  # elementary charge


def _set_axis(ax, xlabel, ylabel, title=None, vlines=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if vlines is not None:
        for x in vlines:
            ax.axvline(x=x)

    return ax


def plot_charge(grid, rho, ax=None, rho_fit=None, **options):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(grid, rho*1e21, label="Numerical")
    if not isinstance(rho_fit, type(None)):
        ax.plot(grid[1:-1], rho_fit[1:-1], label="Analytical")
        ax.legend()

    if ax is None:
        _set_axis(ax, "Position (nm)", r"Charge ($q_e/cm^3$)", **options)
        plt.show()
    else:
        _set_axis(
            ax,
            "Position (nm)",
            r"Charge ($q_e/cm^3$)",
            title="Charge distribution",
            **options
        )
        return ax


def plot_band(grid, band, ax=None, **options):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(grid, band)

    if ax is None:
        _set_axis(ax, "Position (nm)", "Energy (eV)", **options)
        plt.show()
    else:
        _set_axis(
            ax, "Position (nm)", "Energy (eV)", title="Conduction band", **options
        )
        ax.set_title("Conduction band")
        return ax


def plot_wave(grid, psi, energies, ax=None, n_waves=4, **options):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(grid, psi[:, 0]**2, label="$E_0$ = {:f} eV".format(energies[0]))
    for n in np.arange(n_waves - 1) + 1:
        ax.plot(grid, psi[:, n]**2, label="$E_{}$ = {:f} eV".format(n + 1, energies[n]))

    if ax is None:
        _set_axis(ax, "Position (nm)", r"$|\psi|^2 (1/nm)$")
        ax.legend()
        ax.legend(fontsize="xx-small", bbox_to_anchor=(1, 1))
        plt.show()
    else:
        _set_axis(
            ax,
            "Position (nm)",
            r"$|\psi|^2$ ($nm^{-1}$)",
            title=r"Probability",
            **options
        )
        ax.legend(fontsize="xx-small", bbox_to_anchor=(1, 1))
        return ax


def plot_distributions(grid, band, psi, energies, rho, **options):
    fig, ax = plt.subplots(3)

    plot_charge(grid, rho, ax[0], **options)
    plot_band(grid, band, ax[1], **options)
    plot_wave(grid, psi, energies, ax=ax[2], **options)

    fig.tight_layout()
    plt.show()


def plot_charge_density(startV=-1, stopV=1):
    N = 200
    layer = solver.Material("GaAs", 100)
    syst = solver.StackedLayers(0, N, (True, startV), (True, startV), layer)

    N_V = 100

    rho_mid = np.zeros(N_V)

    V = np.linspace(startV, stopV, N_V)

    for i in range(N_V):
        V_0 = np.ones(N) * V[i]
        band = -q_e * V_0
        trans_modes, energies = syst.solve_schrodinger(band)

        rho = syst.solve_charge(trans_modes, energies)
        rho_mid[i] = rho[int(N / 2)]

    plt.plot(V, rho_mid)
    plt.xlabel("$V_0$ (V)")
    plt.ylabel("$n_e$ ($q_e/nm^3$)")
    plt.show()


def plot_wave_band(grid, band, modes, energies):
    # FIGURE
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 7])

    ax = plt.subplot(gs[1])
    ax.plot(grid, band, color='k', label='Band')
    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("Energy (eV)")
    ax.legend(loc=2)

    ax1 = ax.twinx()
    i = 0
    while energies[i] < 0 and i < len(energies):
        ax1.plot(grid, modes[:, i], label="$E_{:d} = {:3f}$".format(i + 1, energies[i]))
        i += 1
    ax1.legend(loc=3)
    ax1.set_ylabel("$\psi$ ($1/\sqrt{nm}$)")

    n_E = 5
    ax2 = plt.subplot(gs[0])
    ax2.scatter(np.zeros(n_E), energies[0:n_E], marker='.', color='k')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    fig.tight_layout()
    plt.show()

def plot_optimize(stacked, options=None):
    if options is None:
        options = {}

    options = dict(options)

    options.setdefault("vlines", None)
    if options["vlines"] is not None:
        options["vlines"] = stacked.L_hj

    band, modes, energies, charge = stacked.solve_optimize()

    plot_distributions(stacked.grid, charge, band, modes, energies, **options)


def plot_energies(grid, band, modes, energies):
    # FIGURE
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 7])

    ax = plt.subplot(gs[1])
    ax.plot(grid, band, color='k', label='Band')
    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("Energy (eV)")
    ax.legend(loc=2)

    ax1 = ax.twinx()
    i = 0
    while True:
        ax1.plot(grid, modes[:, i], label="$E_{:d} = {:3f}$".format(i + 1, energies[i]))
        i += 1
        if not(energies[i] < 0 and i < len(energies)):
            break

    ax1.legend(loc=3)
    ax1.set_ylabel("$\psi$ ($1/\sqrt{nm}$)")

    n_E = 5
    ax2 = plt.subplot(gs[0])
    ax2.scatter(np.zeros(n_E), energies[0:n_E], marker='.', color='k')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    fig.tight_layout()
    plt.show()


def plot_varying_gate(stacked, V_gates, V_surfs, legend=True):
    def sheet_charge(rho, dl):
        return np.sum(rho) * dl

    for V_surf in V_surfs:
        stacked.bound_right = (True, V_surf)
        band = stacked.solve_poisson(np.zeros(stacked.N))
        rho_2d = np.zeros(len(V_gates))
        for i in np.arange(len(V_gates)):
            print(i)
            stacked.bound_left = (True, V_gates[i])
            stacked.bound_right = (True, V_gates[i])
            band, _, _, rho = stacked.solve_snider(band)
            rho_2d[i] = sheet_charge(rho, stacked.dl)

        label = "$V_{surf}$" + " {} V".format(V_surf)
        plt.plot(V_gates, rho_2d, label=label)

    plt.xlabel(r"$V_{0}$ (V)")
    plt.ylabel(r"$\rho_{sheet} (q_e/nm^{2}$)")
    if legend:
        plt.legend()
    plt.show()


def plot_E_n():
    # System
    l = 20  # length of the system (nm)
    N = 500  # number of gridpoints
    N_V = 40
    grid = np.linspace(0, l, N)
    dl = grid[1]

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
    V = np.linspace(-0.799, -0.801, N_V)
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
        plt.plot(V, E_n[j, :], label="n = {}".format(j))

    z = np.polyfit(V, E_n[0, :], 1)
    y = z[1] + z[0] * V
    error = np.sum((y - E_n[0, :]) ** 2) / len(V)
    print("Fit error: ", error)

    plt.plot(V, y, label="Fit")
    plt.xlabel("$V_0$")
    plt.ylabel("$E_n$ (eV)")
    plt.legend()
    plt.savefig("E_n.png")
