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
        VBO = materialproperty[material]['VBO']
        Eg = get_band_gap(material, x)
        CBO = VBO + Eg
    elif material in alloyproperty:
        material1 = alloyproperty[material]["material1"]
        material2 = alloyproperty[material]["material2"]
        VBO1 = materialproperty[material1]['VBO']
        VBO2 = materialproperty[material2]['VBO']
        VBO = x * VBO1 + (1 - x) * VBO2
        Eg = get_band_gap(material, x)
        CBO = VBO + Eg
    else:
        assert False, "Material {} not in database.".format(material)
    return CBO


# MATERIAL PROPERTIES
materialproperty = {
    "GaAs": {
        "m_e": 0.067,
        "epsilonStatic": 12.90,  # dielectric constant
        "Eg": 1.519,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "AlAs": {
        "m_e": 0.15,
        "epsilonStatic": 10.06,
        "Eg": 3.099,
        'VBO': -1.33,
    },
}

# ALLOY PROPERTIES
alloyproperty = {
    "AlGaAs": {
        "bowing_param": 0.37,
        "material1": "AlAs",
        "material2": "GaAs",
    },
    'GaInAs': {
        'bowing_param': 0.477,
        'material1': 'GaAs',
        'material2': 'InAs',
    },
}
