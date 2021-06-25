"""
Database
"""

# PHYSICAL CONSTANTS
k_b = 8.617333262145e-5  # eV/K
epsilon_0 = 0.055263494  # q_e/(V*nm)
m_e = 9.10938 * 10 ** -31  # kg

h_bar = 0.276042828 # eVt (t = 2.3845e-15)
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


def get_m_c(material, x=0):
    return _get_property("m_c", material, x)


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
        "m_c": 0.45,
        "VBO": 0,
        "E_0": 3,
        "epsilonStatic": 4.65,
    },
}

metalproperty = {
    "Al": {
        "m_c": 1,
        "E_f": 11.7,  # eV
        "W": 4.08,  # eV workfunction
    },
}

scproperty = {
    "dummy1": {
        "m_c": 0.067,
        "epsilonStatic": 83.10,  # dielectric constant
        "E_0": 1.37,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "dummy2": {
        "m_c": 0.5,
        "epsilonStatic": 13.10,  # dielectric constant
        "E_0": 1.36,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "GaAs": {
        "m_c": 0.067,
        "epsilonStatic": 13.10,  # dielectric constant
        "E_0": 1.424,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "GaSb": {
        'm_c': .045,
        'epsilonStatic': 15.7,
        'E_0' : 0.81,
        'VBO': -0.03,
        "Delta_0": 0.77,
        "P": 0.9238,
        "g_c": -7.12,
        "gamma_1": 11.8,
        "gamma_2": 4.03,
        "gamma_3": 5.26,
        "kappa": 3.18,
        "q": 0.13,
    },
    "SIGaAs": {
        "m_c": 0.067,
        "epsilonStatic": 13.1,
        "E_0": 1.42,
        "VBO": -0.80,
    },
    "AlAs": {
        "m_c": 0.15,
        "epsilonStatic": 10.06,
        "E_0": 3.099,
        "VBO": -1.33,
    },
    "InAs": {
        "m_c": 0.026,
        "epsilonStatic": 15.15,
        "E_0": 0.417,
        "VBO": -0.59,
        "Delta_0": 0.39,
        "P": 0.9197,
        "g_c": -14.8,
        "gamma_1": 20,
        "gamma_2": 8.5,
        "gamma_3": 9.2,
        "kappa": 7.68,
        "q": 0.4,
    },
    "AlSb": {
        "m_c": 0.14,
        "epsilonStatic": 10.9,
        "E_0": 2.386,
        "VBO": -0.41,
        "Delta_0": 0.676,
        "P": 0.8441,
        "g_c": 0.52,
        "gamma_1": 5.18,
        "gamma_2": 1.19,
        "gamma_3": 1.97,
        "kappa": 0.31,
        "q": 0.07,
    },
    "InSb": {
        "m_c": 0.0135,
        "epsilonStatic": 16.8,
        "E_0": 0.235,
        "VBO": 0,
        "Delta_0": 0.81,
        "P": 0.9641,
        "g_c": -51.56,
        "gamma_1": 34.8,
        "gamma_2": 15.5,
        "gamma_3": 16.5,
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
        "m_c": 0.0092,
    },
    "InSbAs": {
        "material1": "InAs",
        "material2": "InSb",
        "E_0": 0.938,
        "m_c": 0.035,
        "VBO": -0.38,
    },
    "AlInSb": {
        "material1": "AlSb",
        "material2": "InSb",
        "E_0": 0.43,
    },
}
