import math
import os
import warnings
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import kwant
import scipy.linalg as la
import scipy.sparse.linalg as sla
import semicon
import sympy
from scipy import interpolate, optimize
from scipy.linalg import eigh_tridiagonal, lu_factor, lu_solve

from . import database

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
        m_c = np.zeros(self.N)
        doping = np.zeros(self.N)
        band_offset = np.zeros(self.N)
        for i in range(len(layers)):
            where = np.argwhere((grid >= L_hj[i]) * (grid <= L_hj[i + 1]))
            material = layers[i].material
            x = layers[i].x
            epsilon[where] = database.get_dielectric_constant(material, x) * epsilon_0
            m_c[where] = database.get_m_c(material, x)
            band_offset[where] = database.get_band_offset(material, x)
            doping[where] = layers[i].doping

        return L, L_hj, grid, epsilon, m_c, doping, band_offset

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
            self.__epsilon,
            self.__m_c,
            self.doping,
            self.band_offset,
        ) = self._set_properties(layers)

        self.band_offset = self.band_offset - np.amin(self.band_offset)
        self.__CBO = 0

        self.dl = self.grid[0]

        self.DOS = self.m_c / (math.pi * h_bar ** 2)  # Density of States

        self.make_pois_matrix(bound_left, bound_right)
        self.make_system()

    def make_pois_matrix(self, bound_left, bound_right):
        """
        Compute the matrix that is used for solving the Poisson equation.
        Parameters:
        -----------
        bound_left : tuple
            2D tuple of shape (boolean, float) for left/first boundary. True for Dirichlet, False for Neumann.
        bound_right : tuple
            2D tuple of shape (boolean, float) for right/last boundary. True for Dirichlet, False for Neumann.
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
        This is used for the Hamiltonian and subsequently computing the wavefunctions.
        """
        N = self.schrod_stop + 1 - self.schrod_start

        m_c = self.m_c[self.schrod_start : self.schrod_stop + 1]
        m_c_full = np.concatenate(([m_c[0]], m_c, [m_c[-1]]))
        m_c_half = np.convolve(m_c_full, [0.5, 0.5], "valid")
        m_c_imh = m_c_half[0:-1]
        m_c_iph = m_c_half[1::]

        lat = kwant.lattice.chain(self.dl)
        syst = kwant.Builder()

        # ONSITE
        def onsite(site, pot):
            i = site.tag
            t = h_bar ** 2 / (2 * self.dl ** 2) * (1 / m_c_imh[i] + 1 / m_c_iph[i])
            return t + pot[i]

        syst[(lat(x) for x in range(int(N)))] = onsite

        # HOPPING
        for i in np.arange(N - 1) + 1:
            t = h_bar ** 2 / (2 * m_c_imh[i] * self.dl ** 2)
            syst[lat(i), lat(i - 1)] = -t

        # ATTACH LEADS
        left_lead = kwant.Builder(kwant.TranslationalSymmetry((-self.dl,)))
        t = h_bar ** 2 / (2 * m_c_imh[0] * self.dl ** 2)
        left_lead[lat(0)] = 2 * t
        left_lead[lat.neighbors()] = -t
        syst.attach_lead(left_lead)

        right_lead = kwant.Builder(kwant.TranslationalSymmetry((self.dl,)))
        t = h_bar ** 2 / (2 * m_c_iph[-1] * self.dl ** 2)
        right_lead[lat(0)] = 2 * t
        right_lead[lat.neighbors()] = -t
        syst.attach_lead(right_lead)

        self.syst = syst.finalized()

    def solve_charge(self, transverse_modes, energies):
        """
        Solve for the charge distribution.
        Without integrating over k-space.
        Use of solve_charge_dos is recomended.
        """
        inner_product = transverse_modes ** 2
        fd = fermi_dirac_int(0, energies, self.T)

        n_e = np.dot(inner_product, fd) * self.DOS
        rho = -q_e * n_e + q_e * self.doping
        return rho

    def solve_poisson(self, rho):
        """
        Solve the Poisson equation.
        Parameters
        ----------
        rho : float
            N numpy array with charge distribution.

        Returns
        -------
        phi : float
            N numpy array with potential distribution.
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
        Gives the wavefunctions for a given band structure.

        Parameters
        ----------
        band : float
            N numpy array with conduction band, note including the band offset!
        n_modes : int
            number of modes to return.

        Returns
        -------
        full_waves : float
            N x n_modes numpy array with modes in each column.
        energies : float
            n_modes numpy array with energies of each mode.
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
        """
        Find the self-consistent solution of the stack using the scipy root
        optimisation.
        """

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

    def solve_charge_dos(self, phi):
        """
        Compute the charge distribution integrating over k-space for the
        electron density.

        Parameters
        ----------
        phi : float
            N numpy array containing the electrostatic potential distribution.

        Returns
        -------
        rho : float
            N numpy array containing the charge distribution.
        """
        band = -q_e * phi + self.band_offset
        ks = np.linspace(0, 2, 8000)
        dk = ks[1]
        n_e = np.zeros(self.N)
        n_modes = 21

        for k in ks:
            E_k = (h_bar * k) ** 2 / (2 * self.m_c)
            adjusted_band = band + E_k
            psi, energies = self.solve_schrodinger(adjusted_band, n_modes)

            inner_product = psi ** 2
            fd = fermi_dirac(0, energies, self.T)

            n_e += 1 / math.pi * np.dot(inner_product, fd) * k * dk

            # Speed up optimisation
            rel_modes = np.argwhere(fd > 1e-6)  # relevant modes
            if rel_modes.size == 0:
                break
            n_modes = int(rel_modes[-1] + 1)

        rho = -q_e * n_e + self.doping
        return rho

    def matrix_delta_charge(self, phi, option=0):
        """
        Compute the matrix that computes the charge difference in the predictor-
        corrector approach in solve_snider().
        Option 0 uses an approximation.
        Option 1 actually computes the innerproduct, but due to matrix multiplications
            this is relatively slow.
        There is also the case of integrating over k for the density of states.
        """
        band = -q_e * phi + self.band_offset
        n_e = np.zeros(self.N)
        n_modes = 21
        # Approximate DOS
        if option == 0:  # self.T == 0:
            psi, energies = self.solve_schrodinger(band, n_modes)
            inner_product = psi ** 2
            fd = fermi_dirac(0, energies, self.T)

            n_e = self.m_c / math.pi / h_bar ** 2 * np.dot(inner_product, fd)
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
                self.m_c, np.ones(self.N)
            )
            return -q_e * q_e / math.pi / h_bar ** 2 * matrix
        elif self.T == 0:  # Integrate DOS
            ks = np.linspace(0, 1, 1000)
            dk = ks[1]

            for k in ks:
                E_k = (h_bar * k) ** 2 / (2 * self.m_c)
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
        """
        Optimize as is described in the Snider paper. This is a predictor-corrector style approach.

        Parameters
        ----------
        band_init : None/float
            None sets electrostatic potential to zero. N numpy array with band structure (including band offsets)

        Returns
        -------
        band : float
            N numpy array containing the conduction band.
        psi : float
            N x 21 numpy array containing the lowest energy wave functions in each column (psi[:,i]).
        energies : float
            21 numpy array containing the energies of the wavefunctions.
        rho : float
            N numpy array containing the charge distribution.
        """

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

    def check_spurious(self, mat, dat, renormalize=True):
        """
        Check if there are spurious solutions in the bulk of the material.
        Used to deduce wether to renormalize the effective mass in k.p or not.
        """
        a = 0.1
        gamma_0 = 1

        model = semicon.models.ZincBlende(
            components=["foreman"], default_databank="lawaetz"
        )

        if renormalize:
            params = model.parameters(
                mat,
                databank=dat,
            ).renormalize(new_gamma_0=gamma_0)
        else:
            params = model.parameters(
                mat,
                databank=dat,
            )
        # define continuum dispersion function
        continuum = kwant.continuum.lambdify(str(model.hamiltonian), locals=params)

        # define tight-binding dispersion function
        template = kwant.continuum.discretize(model.hamiltonian, grid=a)
        syst = kwant.wraparound.wraparound(template).finalized()
        p = lambda k_x, k_y, k_z: {"k_x": k_x, "k_y": k_y, "k_z": k_z, **params}
        tb = lambda k_x, k_y, k_z: syst.hamiltonian_submatrix(params=p(k_x, k_y, k_z))

        # get dispersions
        N_k = 50
        k = np.linspace(-np.pi / a, np.pi / a, N_k * 2 + 1)
        e = np.array([la.eigvalsh(continuum(k_x=ki, k_y=0, k_z=0)) for ki in k])
        e_tb = np.array([la.eigvalsh(tb(k_x=a * ki, k_y=0, k_z=0)) for ki in k])

        gap_low = e_tb[:, 5][N_k]
        for i in np.arange(e_tb.shape[1] - 2):
            if max(e_tb[:, i]) > gap_low:
                return True

        if min(e_tb[:, -1]) < e_tb[:, -1][N_k]:
            return True
        elif min(e_tb[:, -2]) < e_tb[:, -2][N_k]:
            return True
        return False

    def solve_k_dot_p(
        self,
        sigma=None,
        band=None,
        momenta=np.linspace(-0.12, 0.12, 101),
        savefig=True,
        N_layers="all",
        N_sols=20,
    ):
        """
        Solve the k.p hamiltonian.

        Parameters
        ----------
        sigma : float
            energy around where to look for solutions. Energy of first wavefunction.
        band : np.array(N)
            energy band
        momenta : np.array()
            momenta where to compute k.p
        savefig : boolean
            save the dispersions.
        N_layers : 'all' or array
            what layers to use. If array specify what layers.

        Returns
        -------
        momenta : float
            numpy array containing the momenta over which energies/dispersion are/is computed.
        sorted_levels : float
            numpy array with array of energies for each momentum k in each entry.
        """
        if band is None:
            phi = np.zeros(self.N)
        else:
            phi = -(band - self.band_offset) / q_e

        N_grid = 100
        gamma_0 = 1

        model = semicon.models.ZincBlende(
            components=("foreman",),
            parameter_coords="z",
            default_databank="lawaetz",
        )

        parameters = []
        widths = []
        if N_layers == "all":
            layers = self.layers
        else:
            layers = (self.layers[i] for i in N_layers)
        for layer in layers:
            material = layer.material
            databank = database.get_dict(material, layer.x)

            if databank is False:
                continue

            widths.append(layer.L)
            if self.check_spurious(material, databank):
                print(material, ": non renormalized")
                parameters.append(
                    model.parameters(
                        material,
                        databank=databank,
                        valence_band_offset=databank[material]["VBO"],
                    )
                )
            else:
                print(material, ": renormalized")
                parameters.append(
                    model.parameters(
                        material,
                        databank=databank,
                        valence_band_offset=databank[material]["VBO"],
                    ).renormalize(new_gamma_0=gamma_0)
                )

        grid_spacing = 0.1  # np.sum(widths) / N_grid

        two_deg_params, walls = semicon.misc.two_deg(
            parameters=parameters,
            widths=widths,
            grid_spacing=grid_spacing,
            extra_constants=semicon.parameters.constants,
        )

        xpos = np.arange(0, sum(widths), 0.5)
        # Add potential and shift
        def add_curves(a, b):
            def compute(x):
                return a(x) + b(x)

            return compute

        y1 = two_deg_params["E_v"](xpos)
        y2 = y1 + two_deg_params["E_0"](xpos)
        V_z = interpolate.interp1d(
            self.grid, -q_e * phi - min(y2) + self.CBO, fill_value="extrapolate"
        )  # Adjust such that it matches optimisation
        # two_deg_params["E_v"] = add_curves(two_deg_params["E_v"], V_z)
        print(V_z(xpos))
        # What energy to look for modes
        if sigma is None:
            sigma = min(y2 + V_z(xpos))

        # Make system
        # semicon.misc.plot_2deg_bandedges(two_deg_params, xpos, walls, show_fig=True)
        template = kwant.continuum.discretize(
            model.hamiltonian,
            coords="z",
            grid=grid_spacing,
        )

        shape = lambda site: 0 - grid_spacing / 2 < site.pos[0] < sum(widths)

        syst = kwant.Builder()
        syst.fill(template, shape, (0,))
        syst = syst.finalized()

        # Find and sort solutions
        def eigh_interval(k, levels=N_sols):
            p = {"k_x": k, "k_y": 0, "V": V_z, **two_deg_params}
            ham = syst.hamiltonian_submatrix(params=p, sparse=True)
            ev, evec = sla.eigsh(ham, k=N_sols, sigma=sigma)
            ind = np.argsort(abs(ev))[:levels]
            return ev[ind], evec[:, ind]

        e, psi = eigh_interval(momenta[0])
        sorted_levels = [e]
        for x in momenta[1:]:
            e2, psi2 = eigh_interval(x)
            Q = np.abs(psi.T.conj() @ psi2)  # Overlap matrix
            assignment = optimize.linear_sum_assignment(-Q)[1]
            sorted_levels.append(e2[assignment])
            psi = psi2[:, assignment]

        if savefig:
            plt.figure(figsize=(12, 8))

            plt.plot(momenta, sorted_levels)

            plt.xlim(min(momenta), max(momenta))
            plt.xlabel("$k_x$")
            plt.ylabel("Energy (eV)")

            filename = "figures/dispersion/"
            i = 0
            while True:
                i += 1
                newname = "{}{:d}.png".format(filename, i)
                if os.path.exists(newname):
                    continue
                plt.savefig(newname)
                break

        return momenta, sorted_levels

    def solve_spin_orbit(self, sigma=None, band=None, N_layers="all"):
        """
        Solve k.p using semicon (solve_k_dot_p()) and from this band structure
        extract the shift in k.

        Parameters
        ----------
        energy : float
            energy around where to look for solutions in k.p (energy of first wavefunction)
        band : array (float)
            array with conduction band energy. Default is None.

        Returns
        -------
        shifts : numpy array
            absolute shift of bands below 0 Fermi energy in k-space. Shift is from 0 -> 2xshift is distance
            in k between minima.
        rashba : numpy array
            rashba parameters for all bands below 0 Fermi energy.
        """

        momenta, sorted_levels = self.solve_k_dot_p(
            sigma=sigma, band=band, N_layers=N_layers
        )

        # Find shift in band minima
        N_sols = len(sorted_levels[0])
        ks = np.array([])
        es = np.array([])
        cbands = np.array([])
        rashbas = np.array([])
        for i in np.arange(0, N_sols):
            cband = interpolate.interp1d(
                momenta,
                [e[i] for e in sorted_levels],
                kind="quadratic",
                fill_value="extrapolate",
            )

            # Check conduction band
            p, res, _, _, _ = np.polyfit(momenta, cband(momenta), 2, full=True)
            if p[0] < 0 or res > 1e-4:
                continue

            # Find minima
            x = -p[1] / (2 * p[0])

            # Break if above Fermi energy
            if cband(x) > 3:
                continue
            else:
                ks = np.append(ks, p[2])
                es = np.append(es, cband(x))
                rashba = abs((cband(0.00001) - cband(-0.00001)) / 0.00002)
                print("Residue: ", res)
                print("Rashba: ", rashba)
                rashbas = np.append(rashbas, rashba)
                cbands = np.append(cbands, cband)

        argsort = np.argsort(es)
        ks = ks[argsort].reshape(-1, 2)
        es = es[argsort].reshape(-1, 2)
        rashbas = rashbas[argsort].reshape(-1, 2)
        if len(rashbas) == 0:
            return np.array([0])
        else:
            return rashbas[:, 0]

    # SETTERS AND GETTERS
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
        self.band_offset = self.band_offset - self.__CBO + CBO
        self.__CBO = CBO

    @property
    def m_c(self):
        return self.__m_c

    @m_c.setter
    def m_c(self, m_c):
        self.__m_c = m_c
        self.make_system()

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self.__epsilon = epsilon
        self.make_pois_matrix(self.__bound_left, self.__bound_right)
