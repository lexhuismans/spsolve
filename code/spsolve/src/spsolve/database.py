"""
Database
"""

# PHYSICAL CONSTANTS
k_b = 8.617333262145e-5  # eV/K
epsilon_0 = 0.055263494  # q_e/(V*nm)
m_e = 9.10938 * 10 ** -31  # kg

h_bar = 0.276042828
m_eff = 1.08  # m_e
q_e = 1  # elementary charge


def get_m_e(material, x=0):
    if material in materialproperty:
        m_e = materialproperty[material]["m_e"]
    elif material in alloyproperty:
        material1 = alloyproperty[material]["material1"]
        material2 = alloyproperty[material]["material2"]
        material1_m_e = materialproperty[material1]["m_e"]
        material2_m_e = materialproperty[material2]["m_e"]
        m_e = x * material1_m_e + (1 - x) * material2_m_e
    else:
        assert False, "Material {} not in database".format(material)
    return m_e


def get_dielectric_constant(material, x=0):
    if material in materialproperty:
        eps = materialproperty[material]["epsilonStatic"]
    elif material in alloyproperty:
        material1 = alloyproperty[material]["material1"]
        material2 = alloyproperty[material]["material2"]
        material1_eps = materialproperty[material1]["epsilonStatic"]
        material2_eps = materialproperty[material2]["epsilonStatic"]
        eps = x * material1_eps + (1 - x) * material2_eps
    else:
        assert False, "Material {} not in database".format(material)
    return eps


def get_band_gap(material, x=0):
    if material in materialproperty:
        band_gap = materialproperty[material]["Eg"]
    elif material in alloyproperty:
        material1 = alloyproperty[material]["material1"]
        material2 = alloyproperty[material]["material2"]
        material1_Eg = materialproperty[material1]["Eg"]
        material2_Eg = materialproperty[material2]["Eg"]
        band_gap = (
            x * material1_Eg
            + (1 - x) * material2_Eg
            - x * (1 - x) * alloyproperty[material]["bowing_param"]
        )
    else:
        assert False, "Material {} not in database.".format(material)
    return band_gap


def get_band_offset(material, x=0):
    if material in materialproperty:
        bo = materialproperty[material]["band_offset"]
        Eg = get_band_gap(material, x)
        band_offset = bo * Eg
    elif material in alloyproperty:
        material1 = alloyproperty[material]["material1"]
        material2 = alloyproperty[material]["material2"]
        bo1 = materialproperty[material1]["band_offset"]
        bo2 = materialproperty[material2]["band_offset"]
        bo = x * bo1 + (1 - x) * bo2
        Eg = get_band_gap(material, x)
        band_offset = bo * Eg
    else:
        assert False, "Material {} not in database.".format(material)
    return band_offset


# MATERIAL PROPERTIES
materialproperty = {
    "GaAs": {
        "m_e": 0.067,  # conduction band effective mass (relative to electron mass)
        "m_hh": 0.45,  # heavy hole band effective mass
        "m_lh": 0.087,  # light hole band effective mass
        "epsilonStatic": 12.90,  # dielectric constant
        "Eg": 1.4223,  # 1.42 (ev) band gap
        "Ep": 28.8,  # (eV) k.p matrix element (used for non-parabolicity calculation (Vurgaftman2001)
        "F": -1.94,  # Kane parameter (used for non-parabolicity calculation (Vurgaftman2001)
        "band_offset": 0.65,  # conduction band/valence band offset ratio for GaAs - AlGaAs heterojunctions
        "alpha": 0.605,  # Varschni parameter alpha
        "beta": 204,  # Varschni parameter beta
        "m_e_alpha": 5.3782e18,  # conduction band non-parabolicity variable for linear relation (Nelson approach)
        # Valence band constants
        "delta": 0.28,  # (eV) Spin split-off energy gap
        # below used by aestimo_numpy_h
        "GA1": 6.8,  # luttinger parameter
        "GA2": 1.9,  # luttinger parameter
        "GA3": 2.73,  # luttinger parameter
        "C11": 11.879,  # (GPa) Elastic Constants
        "C12": 5.376,  # (GPa) Elastic Constants
        "a0": 5.6533,  # (A)Lattice constant
        "Ac": -7.17,  # (eV) deformation potentials (Van de Walle formalism)
        "Av": 1.16,  # (eV) deformation potentials (Van de Walle formalism)
        "B": -1.7,  # (eV) shear deformation potential (Van de Walle formalism)
        "TAUN0": 0.1e-7,  # Electron SRH life time
        "TAUP0": 0.1e-7,  # Hole SRH life time
        "mun0": 0.1,  # Electron Mobility in m2/V-s
        "mup0": 0.02,  # Electron Mobility in m2/V-s
        "Cn0": 2.8e-31,  # generation recombination model parameters [cm**6/s]
        "Cp0": 2.8e-32,  # generation recombination model parameters [cm**6/s]
        "BETAN": 2.0,  # Parameter in calculatation of the Field Dependant Mobility
        "BETAP": 1.0,  # Parameter in calculatation of the Field Dependant Mobility
        "VSATN": 3e5,  # Saturation Velocity of Electrons
        "VSATP": 6e5,  # Saturation Velocity of Holes
        "AVb_E": -6.92,  # Average Valence Band Energy or the absolute energy level
    },
    "AlAs": {
        "m_e": 0.15,
        "m_hh": 0.51,
        "m_lh": 0.18,
        "epsilonStatic": 10.06,
        "Eg": 3.0,  # 2.980,
        "Ep": 21.1,
        "F": -0.48,
        "band_offset": 0.53,
        "alpha": 0.605,
        "beta": 204,
        "m_e_alpha": 0.0,
        "GA1": 3.45,
        "GA2": 0.68,
        "GA3": 1.29,
        "C11": 11.879,
        "C12": 5.376,
        "a0": 5.66,
        "Ac": -5.64,
        "Av": 2.47,
        "B": -1.5,
        "delta": 0.28,
        "TAUN0": 0.1e-6,
        "TAUP0": 0.1e-6,
        "mun0": 0.15,
        "mup0": 0.1,
        "Cn0": 2.8e-31,  # generation recombination model parameters [cm**6/s]
        "Cp0": 2.8e-32,  # generation recombination model parameters [cm**6/s]
        "BETAN": 2.0,
        "BETAP": 1.0,  # Parameter in calculatation of the Field Dependant Mobility
        "VSATN": 3e5,  # Saturation Velocity of Electrons
        "VSATP": 6e5,  # Saturation Velocity of Holes
        "AVb_E": -7.49,  # Average Valence Band Energy or the absolute energy level
    },
}

# ALLOY PROPERTIES
alloyproperty = {
    "AlGaAs": {
        "bowing_param": 0.37,
        "band_offset": 0.65,
        "m_e_alpha": 5.3782e18,
        "delta_bowing_param": 0.0,
        "a0_sub": 5.6533,
        "material1": "AlAs",
        "material2": "GaAs",
        "TAUN0": 0.1e-6,
        "TAUP0": 0.1e-6,
        "mun0": 0.15,
        "mup0": 0.1,
        "Cn0": 2.8e-31,  # generation recombination model parameters [cm**6/s]
        "Cp0": 2.8e-32,  # generation recombination model parameters [cm**6/s]
        "BETAN": 2.0,
        "BETAP": 1.0,  # Parameter in calculatation of the Field Dependant Mobility
        "VSATN": 3e5,  # Saturation Velocity of Electrons
        "VSATP": 6e5,  # Saturation Velocity of Holes
        "AVb_E": -2.1,  # Average Valence Band Energy or the absolute energy level
    }
}
