import math
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import kwant
import scipy.sparse.linalg as sla
import semicon
import sympy
from scipy import interpolate, optimize
from scipy.linalg import eigh_tridiagonal, lu_factor, lu_solve

from . import database, plotter

# Physical constants
k_b = database.k_b
epsilon_0 = database.epsilon_0  # q_e/(V*nm)
m_e = database.m_e  # kg
h_bar = database.h_bar
q_e = database.q_e  # elementary charge

Material = namedtuple("Material", ["material", "L", "x", "doping"], defaults=[0, 0])


def fermi_dirac(E_f, E, T):
    if T == 0:
        return E < E_f
    else:
        return 1 / (1 + np.exp((E - E_f) / k_b / T))


def fermi_dirac_int(E_F, E, T):
    """
    Computes the integral of the Fermi-Dirac distribution. Trapezoidal integration.
    (As is needed forthe number of electrons in a subband (n) with a two dimensional
    DOS)
    """
    if T == 0:
        below_fermi = E < E_F
        return (E_F - E) * below_fermi
    else:
        fd = fermi_dirac(E_F, E, T)
        d = np.diff(E)
        integral = np.zeros(len(d) + 1)

        # Compute integrals
        for i in np.flip(np.arange(len(fd) - 1)):
            integral[i] = d[i] * (fd[i] + fd[i + 1]) / 2 + integral[i + 1]
        return integral


def fermi_dirac_der(E_F, E, T):
    if T > 200:
        exp = np.exp((E - E_F) / (k_b * T))
        return -exp / (1 + exp) ** 2 * 1 / (k_b * T)
    else:
        T = 200
        exp = np.exp((E - E_F) / (k_b * T))
        if not np.isfinite(exp).all():
            return np.zeros(len(E))
        else:
            fdder = -exp / (1 + exp) ** 2 * 1 / (k_b * T)
            if not np.isfinite(fdder).all():
                return np.zeros(len(E))
            else:
                return fdder


class StackedLayers:
    def _set_properties(self, layers):
        L = 0  # Total length (nm)
        L_hj = np.array([0])
        for layer in layers:
            L += layer.L
            L_hj = np.append(L_hj, L)

        grid = np.linspace(0, L, num=self.N + 2)[1:-1]

        epsilon = np.zeros(self.N)
        m_e = np.zeros(self.N)
        doping = np.zeros(self.N)
        band_offset = np.zeros(self.N)
        for i in range(len(layers)):
            where = np.argwhere((grid >= L_hj[i]) * (grid <= L_hj[i + 1]))
            material = layers[i].material
            x = layers[i].x
            epsilon[where] = database.get_dielectric_constant(material, x) * epsilon_0
            m_e[where] = database.get_m_e(material, x)
            band_offset[where] = database.get_band_offset(material, x)
            doping[where] = layers[i].doping

        return L, L_hj, grid, epsilon, m_e, doping, band_offset

    def __init__(self, T, N, bound_left, bound_right, *layers):
        self.layers = layers  # Layers (tuple)
        self.T = T
        self.N = N
        self.schrod_start = 0
        self.schrod_stop = N - 1

        # PROPERTIES
        (
            self.L,
            self.L_hj,
            self.grid,
            self.epsilon,
            self.m_e,
            self.doping,
            self.band_offset,
        ) = self._set_properties(layers)

        self.band_offset = self.band_offset - np.amin(self.band_offset)

        self.dl = self.grid[0]

        self.DOS = self.m_e / (math.pi * h_bar ** 2)  # Density of States

        self.make_pois_matrix(bound_left, bound_right)
        self.make_system()

    def make_pois_matrix(self, bound_left, bound_right):
        """
        Compute the matrix that is used for solving the Poisson equation.
        """
        epsilon_full = np.concatenate(
            ([self.epsilon[0]], self.epsilon, [self.epsilon[-1]])
        )  # Include edges
        epsilon_half = np.convolve(epsilon_full, [0.5, 0.5], "valid")
        epsilon_imh = epsilon_half[0:-1]  # epsilon^i-1/2
        epsilon_iph = epsilon_half[1::]  # epsilon^i+1/2

        km1 = epsilon_imh
        k = -(epsilon_imh + epsilon_iph)
        kp1 = epsilon_iph

        prefac = -1 / self.dl ** 2

        self.pois_matrix = prefac * (
            np.diag(km1[1::], k=-1) + np.diag(k, k=0) + np.diag(kp1[0:-1], k=1)
        )

        self.bound_left = bound_left
        self.bound_right = bound_right

    def make_system(self):
        """
        Build a kwant system which is a one dimensional chain.
        template = kwant.continuum.discretize('k_x * A(x) * k_x + V(x)')
        print(template)
        """
        N = self.schrod_stop + 1 - self.schrod_start

        m_e = self.m_e[self.schrod_start : self.schrod_stop + 1]
        m_e_full = np.concatenate(([m_e[0]], m_e, [m_e[-1]]))
        m_e_half = np.convolve(m_e_full, [0.5, 0.5], "valid")
        m_e_imh = m_e_half[0:-1]
        m_e_iph = m_e_half[1::]

        lat = kwant.lattice.chain(self.dl)
        syst = kwant.Builder()

        # ONSITE
        def onsite(site, pot):
            i = site.tag
            t = h_bar ** 2 / (2 * self.dl ** 2) * (1 / m_e_imh[i] + 1 / m_e_iph[i])
            return t + pot[i]

        syst[(lat(x) for x in range(int(N)))] = onsite
        # HOPPING
        for i in np.arange(N - 1) + 1:
            t = h_bar ** 2 / (2 * m_e_imh[i] * self.dl ** 2)
            syst[lat(i), lat(i - 1)] = -t

        # ATTACH LEADS
        left_lead = kwant.Builder(kwant.TranslationalSymmetry((-self.dl,)))
        t = h_bar ** 2 / (2 * m_e_imh[0] * self.dl ** 2)
        left_lead[lat(0)] = 2 * t
        left_lead[lat.neighbors()] = -t
        syst.attach_lead(left_lead)

        right_lead = kwant.Builder(kwant.TranslationalSymmetry((self.dl,)))
        t = h_bar ** 2 / (2 * m_e_iph[-1] * self.dl ** 2)
        right_lead[lat(0)] = 2 * t
        right_lead[lat.neighbors()] = -t
        syst.attach_lead(right_lead)

        self.syst = syst.finalized()

    def solve_charge(self, transverse_modes, energies):
        """
        Solve for the charge distribution.
        """
        inner_product = transverse_modes ** 2
        fd = fermi_dirac_int(0, energies, self.T)

        n_e = np.dot(inner_product, fd) * self.DOS
        rho = -q_e * n_e + q_e * self.doping
        return rho

    def solve_poisson(self, rho):
        """
        Solve the Poisson equation non-uniform mesh.
        """
        adjusted_rho = rho.copy()

        # --------------------BOUNDARIES----------------------
        if self.bound_left[0]:
            # Dirichlet
            adjusted_rho[0] += self.epsilon[0] * (self.bound_left[1]) / self.dl ** 2
        else:
            # Neumann
            adjusted_rho[0] += -2 * self.bound_left[1] * self.epsilon[0] / self.dl

        if self.bound_right[0]:
            # Dirichlet
            adjusted_rho[-1] += self.epsilon[-1] * (self.bound_right[1]) / self.dl ** 2
        else:
            # Neumann
            adjusted_rho[-1] += 2 * self.bound_right[1] * self.epsilon[-1] / self.dl

        # ---------------------SOLVE--------------------------
        phi = lu_solve(self.pois_matrix_lu_piv, adjusted_rho)
        # band = -q_e * phi + self.band_offset + self.CBO
        return phi

    def solve_schrodinger(self, band, n_modes=21):
        """
        Gives the wavefunctions for a given potential distribution.
        """
        band = band[self.schrod_start : self.schrod_stop + 1]
        ham = self.syst.hamiltonian_submatrix(sparse=False, params=dict(pot=band))
        diag = np.real(ham.diagonal())
        off_diag = np.real(ham.diagonal(1))

        energies, transverse_modes = eigh_tridiagonal(
            diag, off_diag, select="i", select_range=(0, n_modes - 1)
        )

        transverse_modes = transverse_modes / math.sqrt(
            self.dl
        )  # Every column is a wavefunction

        # Add zeros to modes
        full_waves = np.zeros((self.N, transverse_modes.shape[1]))
        full_waves[self.schrod_start : self.schrod_stop + 1, :] = transverse_modes
        return full_waves, energies

    def solve_optimize(self, band_init=None):
        def self_consistent(phi):
            phi_old = phi.copy()

            rho = self.solve_charge_dos(phi)
            phi = self.solve_poisson(rho)

            # Compute error
            diff = phi_old - phi
            # plotter.plot_distributions(self.grid, band, psi, energies, rho)
            print(np.sum(np.abs(diff)))
            return diff

        if band_init is None:
            phi_init = self.solve_poisson(np.zeros(self.N))
        else:
            phi_init = -(band_init - self.band_offset) / q_e

        optim_result = optimize.root(
            self_consistent,
            phi_init,
            method="anderson",
            tol=1e-2,  # , options=dict(maxiter=3)
        )

        phi = optim_result.x
        rho = self.solve_charge_dos(phi)
        band = -q_e * phi + self.band_offset
        psi, energies = self.solve_schrodinger(band)
        return band, psi, energies, rho

    def _solve_charge(self, phi, fd_func):
        band = -q_e * phi + self.band_offset
        ks = np.linspace(0, 2, 8000)
        dk = ks[1]
        n_e = np.zeros(self.N)
        n_modes = 21

        for k in ks:
            E_k = (h_bar * k) ** 2 / (2 * self.m_e)
            adjusted_band = band + E_k
            psi, energies = self.solve_schrodinger(adjusted_band, n_modes)

            inner_product = psi ** 2
            fd = fd_func(0, energies, self.T)

            n_e += 1 / math.pi * np.dot(inner_product, fd) * k * dk

            # Speed up optimisation
            rel_modes = np.argwhere(fd > 1e-6)  # relevant modes
            if rel_modes.size == 0:
                break
            n_modes = int(rel_modes[-1] + 1)

        return -q_e * n_e + self.doping

    def solve_charge_dos(self, phi):
        return self._solve_charge(phi, fermi_dirac)

    def matrix_delta_charge(self, phi, option=0):
        band = -q_e * phi + self.band_offset
        n_e = np.zeros(self.N)
        n_modes = 21
        # Approximate DOS
        if option == 0:  # self.T == 0:
            psi, energies = self.solve_schrodinger(band, n_modes)
            inner_product = psi ** 2
            fd = fermi_dirac(0, energies, self.T)

            n_e = self.m_e / math.pi / h_bar ** 2 * np.dot(inner_product, fd)
            return np.diag(-q_e * q_e * n_e)
        elif option == 1:
            psi, energies = self.solve_schrodinger(band, n_modes)
            inner_product = psi ** 2
            fd = fermi_dirac(0, energies, self.T)

            fd_where = fd > 1e-5

            fd_modes = inner_product[:, fd_where] * np.outer(
                np.ones(self.N), np.sqrt(fd[fd_where])
            )
            matrix = np.matmul(fd_modes, fd_modes.T) * np.outer(
                self.m_e, np.ones(self.N)
            )
            return -q_e * q_e / math.pi / h_bar ** 2 * matrix
        elif self.T == 0:  # Integrate DOS
            ks = np.linspace(0, 1, 1000)
            dk = ks[1]

            for k in ks:
                E_k = (h_bar * k) ** 2 / (2 * self.m_e)
                adjusted_band = band + E_k
                psi, energies = self.solve_schrodinger(adjusted_band, n_modes)

                inner_product = psi ** 2
                fd = fermi_dirac_der(0, energies, self.T)

                n_e += 1 / math.pi * np.dot(inner_product, fd) * k * dk

                # Speed up optimisation
                rel_modes = np.argwhere(fd > 1e-6)  # relevant modes
                if rel_modes.size == 0:
                    break
                n_modes = int(rel_modes[-1] + 1)
            return q_e * n_e

    def solve_snider(self, band_init=None):
        def adjust_rho(rho):
            adj_rho = rho.copy()
            # Adjust rho for boundaries
            if self.bound_left[0]:
                # Dirichlet
                adj_rho[0] += self.epsilon[0] * (self.bound_left[1]) / self.dl ** 2
            else:
                # Neumann
                adj_rho[0] += -2 * self.bound_left[1] * self.epsilon[0] / self.dl

            if self.bound_right[0]:
                # Dirichlet
                adj_rho[-1] += self.epsilon[-1] * (self.bound_right[1]) / self.dl ** 2
            else:
                # Neumann
                adj_rho[-1] += 2 * self.bound_right[1] * self.epsilon[-1] / self.dl
            return adj_rho

        error = np.ones(self.N)
        tolerance = 1e-5

        rho_prev = self.doping

        phi = np.zeros(self.N)
        if band_init is None:
            phi = self.solve_poisson(rho_prev)
        else:
            phi = -(band_init - self.band_offset) / q_e
        delta_phi = np.zeros(self.N)

        error_prev = 10000
        option = 0
        while True:
            rho = self.solve_charge_dos(phi)
            trial_error = self.pois_matrix.dot(phi) - adjust_rho(rho)
            error = np.abs(trial_error)
            # print(np.sum(np.abs(error)))
            if np.all(error < tolerance):
                break
            elif error_prev < np.sum(np.abs(error)):
                print("option: 1")
                option = 1
                # print("Oscilations -> start SciPy optimisation")
                # return self.solve_optimize(-q_e * phi + self.band_offset)
            error_prev = np.sum(np.abs(error))

            matrix = -self.pois_matrix + self.matrix_delta_charge(phi, option)
            delta_phi = np.linalg.solve(matrix, trial_error)

            phi = delta_phi + phi

        band = -q_e * phi + self.band_offset
        psi, energies = self.solve_schrodinger(band)
        return band, psi, energies, rho

    def solve_k_dot_p(self, band=None):
        if band is None:
            phi = np.zeros(self.N)
        else:
            phi = -(band - self.band_offset) / q_e

        N_grid = 80
        gamma_0 = 1

        model = semicon.models.ZincBlende(
            components=("foreman",),
            parameter_coords="z",
            default_databank="lawaetz",
        )

        parameters = []
        widths = []
        for layer in self.layers:
            material = layer.material
            databank = database.get_dict(material, layer.x)

            if databank is False:
                continue

            widths.append(layer.L)

            parameters.append(
                model.parameters(
                    material,
                    databank=databank,
                    valence_band_offset=databank[material]["VBO"] + 0.59,
                ).renormalize(new_gamma_0=gamma_0)
            )

        grid_spacing = np.sum(widths) / N_grid

        two_deg_params, walls = semicon.misc.two_deg(
            parameters=parameters,
            widths=widths,
            grid_spacing=grid_spacing,
            extra_constants=semicon.parameters.constants,
        )

        xpos = np.arange(-2 * grid_spacing, sum(widths) + 2 * grid_spacing, 0.5)
        semicon.misc.plot_2deg_bandedges(two_deg_params, xpos, walls, show_fig=True)

        template = kwant.continuum.discretize(
            model.hamiltonian + sympy.diag(*[" + V(z)"] * 8),
            coords="z",
            grid=grid_spacing,
        )

        shape = lambda site: 0 - grid_spacing / 2 < site.pos[0] < sum(widths)

        syst = kwant.Builder()
        syst.fill(template, shape, (0,))
        syst = syst.finalized()

        momenta = np.linspace(-0.45, 0.45, 101)
        V_z = interpolate.interp1d(self.grid, -q_e * phi, fill_value="extrapolate")

        energies = []
        for k in momenta:
            p = {"k_x": k, "k_y": 0, "V": V_z, **two_deg_params}
            ham = syst.hamiltonian_submatrix(params=p, sparse=False)
            ev, evec = sla.eigsh(ham, k=20, sigma=0.52)
            energies.append(ev)

        plt.figure(figsize=(12, 8))

        plt.plot(momenta, np.sort(energies))

        plt.xlim(min(momenta), max(momenta))
        plt.ylim(0.36, 0.72)
        plt.show()

    @property
    def bound_left(self):
        return self.__bound_left

    @bound_left.setter
    def bound_left(self, bound):
        prefac = -1 / self.dl ** 2
        if bound[0] is False:
            self.pois_matrix[0, :2] = prefac * np.array([-2, 1]) * self.epsilon[0]
            self.pois_matrix[0, 1] += prefac * self.epsilon[0]
        else:
            self.pois_matrix[0, :2] = prefac * np.array([-2, 1]) * self.epsilon[0]

        lu, piv = lu_factor(self.pois_matrix)
        self.pois_matrix_lu_piv = (lu, piv)
        self.__bound_left = bound

    @property
    def bound_right(self):
        return self.__bound_right

    @bound_right.setter
    def bound_right(self, bound):
        prefac = -1 / self.dl ** 2
        if bound[0] is False:
            self.pois_matrix[-1, -2:] = prefac * np.array([1, -2]) * self.epsilon[-1]
            self.pois_matrix[-1, -2] += prefac * self.epsilon[-1]
        else:
            self.pois_matrix[-1, -2:] = prefac * np.array([1, -2]) * self.epsilon[-1]

        lu, piv = lu_factor(self.pois_matrix)

        self.pois_matrix_lu_piv = (lu, piv)
        self.__bound_right = bound

    @property
    def schrod_where(self):
        return self.__schrod_where

    @schrod_where.setter
    def schrod_where(self, bounds):
        try:
            x0 = np.argwhere(self.grid >= bounds[0])[0]
        except IndexError:
            x0 = 0
        try:
            x1 = np.argwhere(self.grid <= bounds[1])[-1]
        except IndexError:
            x1 = self.N - 1

        self.schrod_start = int(x0)
        self.schrod_stop = int(x1)
        self.__schrod_where = (self.grid[x0], self.grid[x1])
        self.make_system()

    @property
    def CBO(self):
        return self.__CBO

    @CBO.setter
    def CBO(self, CBO):
        self.band_offset = self.band_offset + CBO
        self.__CBO = CBO


# -----------------------Non Class Code---------------------------


def comp_schrod_matrix(phi, delta_l, m_eff=1.08):
    """
    Builds the matrix for solving the Schrodinger equation.

    Parameters
    ----------
    phi : float
        vector phi containing potential values for each gridpoint.
    delta_l : float
        distance between each gridpoint.
    m_eff : float
        effective mass of electron in m_e.

    Returns
    -------
    schrod_matrix : float
        numpy matrix used in Schrodinger equation for solving wavefunction.
    """
    N = phi.size
    pre_fac = h_bar ** 2 / (2 * m_eff * delta_l ** 2)
    """
    schrod_matrix = (np.eye(N, k=-1)+np.eye(N, k=1))
    np.fill_diagonal(schrod_matrix,((q_e*phi/pre_fac-2)))
    schrod_matrix = -pre_fac*schrod_matrix
    """
    schrod_matrix = (np.eye(N, k=-1) + np.eye(N, k=1)) - 2 * np.eye(N)
    schrod_matrix = -pre_fac * schrod_matrix

    potential_matrix = np.zeros((N, N))
    diagonal = phi + np.append(0, phi[0:-1])
    np.fill_diagonal(potential_matrix, diagonal)
    potential_matrix = (
        potential_matrix
        + np.eye(N, k=1) * np.append(0, -phi[0:-1])
        + np.eye(N, k=-1) * np.append(-phi[0:-1], 0)
    )

    schrod_matrix = schrod_matrix + potential_matrix
    return schrod_matrix


def make_system(dl, L, m_eff):
    """
    Build a kwant system which is a one dimensional chain.

    Parameters
    ----------
    dl : float
        distance between each gridpoint.
    L : int
        length of system in nm.
    m_eff : float
        effective mass of electron in m_e.

    Returns
    -------
    syst : kwant.Builder
        finalized system (kwant).
    """
    t = h_bar ** 2 / (2 * m_eff * dl ** 2)

    lat = kwant.lattice.chain(dl)
    syst = kwant.Builder()

    def onsite(site, pot):
        return 2 * t - q_e * pot[site.tag]

    syst[(lat(x) for x in range(int(L)))] = onsite
    syst[lat.neighbors()] = -t

    # Attach leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-dl,)))
    lead[lat(0)] = 2 * t
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst.finalized()


def comp_pois_matrix(dl, N, bound_one_dirich=True, bound_two_dirich=True):
    """
    Builds the matrix for solving the Poisson equation

    Parameters
    ----------
    dl : float
        distance between each gridpoint.
    N : int
        number of gridpoints.
    bound_one_dirich : boolean
        True if first boundary is Dirichlet, False for Neumann.
    bound_two_dirich : boolean
        True if second boundary is Dirichlet, False for Neumann.

    Returns
    -------
    pois_matrix : float
        numpy matrix used in Poisson equation for solving potential.
    """
    pois_matrix = np.eye(N, k=-1) + np.eye(N, k=1) - 2 * np.eye(N)

    # Change according to boundary conditions
    # First boundary
    if bound_one_dirich:
        pois_matrix[0, 0:2] = [1, 0]
    else:
        pois_matrix[0, 0:2] = [-2, 2]
    # Second boundary
    if bound_two_dirich:
        pois_matrix[-1, -2:] = [0, 1]
    else:
        pois_matrix[-1, -2:] = [2, -2]

    pois_matrix = -epsilon_0 / dl ** 2 * pois_matrix
    return pois_matrix


def solve_poisson(pois_matrix, rho, dl, alpha, beta):
    """
    Solves the discrete Poisson equation.

    Parameters
    ----------
    pois_matrix : float
        Poisson matrix as given by comp_pois_matrix
    rho : float
        charge distribution vector
    dl : float
        distance between gridpoints
    alpha : float
        left boundary value
    beta : float
        right boundary value

    Returns
    -------
    phi : float
        potential vector
    """
    adjusted_rho = rho.copy()
    # Adjust charge distribution according to boundary condition
    # First boundary
    if pois_matrix[0, 1] == 0:
        adjusted_rho[0] = -alpha * epsilon_0 / dl ** 2
    else:
        adjusted_rho[0] = adjusted_rho[0] + 2 * alpha * epsilon_0 / dl
    # Second boundary
    if pois_matrix[-1, -2] == 0:
        adjusted_rho[-1] = -beta * epsilon_0 / dl ** 2
    else:
        adjusted_rho[-1] = adjusted_rho[-1] - 2 * beta * epsilon_0 / dl

    # Solve equation
    phi = np.linalg.solve(pois_matrix, adjusted_rho)
    return phi


def solve_schrod(phi, dl):
    """
    Solves the discrete Schrodinger equation.

    Parameters
    ----------
    phi : float
        potential vector.
    dl : float
        distance between gridpoints.

    Returns
    -------
    energies : float
        eigen energies of the solution (eV).
    psi : float
        wavefunctions (eigenvectors of the solution). 2D array with
        wavefunction for each i, psi[:,i].
    """
    schrod_matrix = comp_schrod_matrix(phi, dl)
    energies, psi = np.linalg.eig(schrod_matrix)

    # Sort from lowest to highest energy
    sort_index = np.argsort(energies)
    energies = energies[sort_index]
    psi = psi[:, sort_index] / math.sqrt(dl)  # Every column is a wavefunction
    return energies, psi


def solve_syst(syst, phi, dl):
    """
    Gives the wavefunctions for a given potential distribution.

    Parameters
    ----------
    syst : kwant.Builder.finalized
        finalized system.
    phi : float
        array with potential for each gridpoint.
    dl : float
        distance between gridpoints.

    Returns
    -------
    energies : float
        energies of all the wavefunctions.
    psi : float
        2D array with wavefunction for each i, psi[:, i].
    """
    V = q_e * phi
    ham = syst.hamiltonian_submatrix(sparse=False, params=dict(pot=V))
    # print('Hamiltonian: \n', ham)
    # t0 = time.time()
    energies, psi = eigh(ham)
    # t1 = time.time()
    # print('Solve ham: ', t1-t0)
    # print('Energies: \n', energies)

    # Sort from lowest to highest energy
    sort_index = np.argsort(energies)
    energies = energies[sort_index]
    psi = np.real(psi[:, sort_index]) / math.sqrt(dl)  # Every column is a wavefunction

    return np.real(energies), psi


def solve_charge_dist(psi, energies, Ef, T=0, N_D=None):
    """
    Solve the for the charge distribution given the wavefunctions.

    Parameters
    ----------
    psi : float
        2D array with wavefunction for each i, psi[:,i].
    energies : float
        1D array with the energies corresponding to the wavefunction psi[:, i].
    Ef : float
        Fermi energy.
    T : float
        temperature (Kelvin).
    N_D : float
        1D array with doping concentrations.

    Returns
    -------
    rho : float
        1D array with charge distribution.
    """
    DOS = m_eff / (math.pi * h_bar ** 2)  # Density of States
    if T == 0:
        # Remove wavefunctions with energy above Fermi energy.
        psi = psi[:, np.argwhere(energies < Ef)]
        energies = energies[np.argwhere(energies < Ef)]

        # Compute charge density
        inner_product = np.squeeze(psi ** 2, axis=2)
        N_elec = DOS * (Ef - energies)

        rho = -q_e * np.dot(inner_product, N_elec)

        # Add doping
        if not isinstance(N_D, type(None)):
            rho = rho + q_e * N_D
    else:
        inner_product = psi ** 2
        N_elec = DOS * fermi_dirac_int(Ef, energies, T)
        rho = -q_e * np.dot(inner_product, N_elec)

        # Add doping
        if not isinstance(N_D, type(None)):
            rho = rho + q_e * N_D

    return rho
