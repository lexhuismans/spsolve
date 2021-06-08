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


def get_dict(material, x=0):
    params = {}
    if material in scproperty:
        properties = scproperty[material].keys()
    elif material in alloyproperty:
        properties = scproperty[alloyproperty[material]["material1"]].keys()
    else:
        return False

    for property in properties:
        params[property] = _get_property(property, material, x)

    return {material: params}


def get_m_e(material, x=0):
    return _get_property("m_e", material, x)


def get_dielectric_constant(material, x=0):
    return _get_property("epsilonStatic", material, x)


def get_band_gap(material, x=0):
    return _get_property("E_0", material, x)


def get_band_offset(material, x=0):
    VBO = _get_property("VBO", material, x)
    E_0 = _get_property("E_0", material, x)
    return VBO + E_0


def _get_property(property, material, x=0):
    if material in scproperty:
        return scproperty[material][property]
    elif material in alloyproperty:
        material1 = alloyproperty[material]["material1"]
        material2 = alloyproperty[material]["material2"]

        prop1 = scproperty[material1][property]
        prop2 = scproperty[material2][property]

        if property in alloyproperty[material]:
            bowing = alloyproperty[material][property]
        else:
            bowing = 0
        return x * prop1 + (1 - x) * prop2 - x * (1 - x) * bowing
    elif material in metalproperty:
        return metalproperty[material][property]
    elif material in dielectricproperty:
        return dielectricproperty[material][property]
    else:
        raise print("Material not found :(")


# MATERIAL PROPERTIES
dielectricproperty = {
    "SiNx": {
        "m_e": 0.45,
        "VBO": 0,
        "E_0": 3,
        "epsilonStatic": 4.65,
    },
}

metalproperty = {
    "Al": {
        "m_e": 1,
        "E_f": 11.7,  # eV
        "W": 4.08,  # eV workfunction
    },
}

scproperty = {
    "dummy1": {
        "m_e": 0.067,
        "epsilonStatic": 83.10,  # dielectric constant
        "E_0": 1.37,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "dummy2": {
        "m_e": 0.5,
        "epsilonStatic": 13.10,  # dielectric constant
        "E_0": 1.36,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "GaAs": {
        "m_e": 0.067,
        "epsilonStatic": 13.10,  # dielectric constant
        "E_0": 1.424,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "GaSb": {
        'm_e': .039,
        'epsilonStatic': 15.7,
        'E_0' : 0.81,
        'VBO': -0.03,
        "Delta_0": 0.77,
        "P": 0.9238,
        "m_c": 0.045,
        "g_c": -7.12,
        "gamma_1": 11.8,
        "gamma_2": 4.03,
        "gamma_3": 5.26,
        "kappa": 3.18,
        "q": 0.13,
    },
    "SIGaAs": {
        "m_e": 0.067,
        "epsilonStatic": 13.1,
        "E_0": 1.42,
        "VBO": -0.80,
    },
    "AlAs": {
        "m_e": 0.15,
        "epsilonStatic": 10.06,
        "E_0": 3.099,
        "VBO": -1.33,
    },
    "InAs": {
        "m_e": 0.027,
        "epsilonStatic": 15.15,
        "E_0": 0.417,
        "VBO": -0.59,
        "Delta_0": 0.38,
        "P": 0.9197,
        "m_c": 0.023,
        "g_c": -14.8,
        "gamma_1": 19.67,
        "gamma_2": 8.37,
        "gamma_3": 9.29,
        "kappa": 7.68,
        "q": 0.04,
    },
    "AlSb": {
        "m_e": 0.14,
        "epsilonStatic": 10.9,
        "E_0": 2.386,
        "VBO": -0.41,
        "Delta_0": 0.75,
        "P": 0.8441,
        "m_c": 0.18,
        "g_c": 0.52,
        "gamma_1": 4.15,
        "gamma_2": 1.01,
        "gamma_3": 1.75,
        "kappa": 0.31,
        "q": 0.07,
    },
    "InSb": {
        "m_e": 0.018,
        "epsilonStatic": 16.8,
        "E_0": 0.235,
        "VBO": 0,
        "Delta_0": 0.81,
        "P": 0.9641,
        "m_c": 0.0139,
        "g_c": -51.56,
        "gamma_1": 37.1,
        "gamma_2": 16.5,
        "gamma_3": 17.7,
        "kappa": 15.6,
        "q": 0.39,
    },
}

# ALLOY PROPERTIES
alloyproperty = {
    "AlGaAs": {
        "material1": "AlAs",
        "material2": "GaAs",
        "E_0": 0.37,
    },
    "GaInAs": {
        "material1": "GaAs",
        "material2": "InAs",
        "E_0": 0.415,
        "m_e": 0.0092,
    },
    "InSbAs": {
        "material1": "InAs",
        "material2": "InSb",
        "E_0": 0.067,
        "m_e": 0.035,
    },
    "AlInSb": {
        "material1": "AlSb",
        "material2": "InSb",
        "E_0": 0.43,
    },
}
