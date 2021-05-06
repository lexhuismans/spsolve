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
        print('Material ' + material + ' not found :(')


def get_m_e(material, x=0):
    return _get_property("m_e", material, x)


def get_dielectric_constant(material, x=0):
    return _get_property("epsilonStatic", material, x)


def get_band_gap(material, x=0):
    return _get_property("Eg", material, x)


def get_band_offset(material, x=0):
    VBO = _get_property("VBO", material, x)
    Eg = _get_property("Eg", material, x)
    return VBO + Eg

# MATERIAL PROPERTIES
dielectricproperty = {
    'SiNx':{
        'm_e': .45,
        'VBO': 0,
        'Eg': 3,
        'epsilonStatic': 4.65,
    },
}

metalproperty = {
    'Al':{
        'm_e': 1,
        'E_f': 11.7, # eV
        'W': 4.08, # eV workfunction
    },
}

# MATERIAL PROPERTIES
scproperty = {
    "GaAs": {
        "m_e": 0.067,
        "epsilonStatic": 12.90,  # dielectric constant
        "Eg": 1.519,  # (eV) Fundamental energy gap
        "VBO": -0.80,  # (eV)
    },
    "AlAs": {"m_e": 0.15, "epsilonStatic": 10.06, "Eg": 3.099, "VBO": -1.33,},
    "InAs": {"m_e": 0.027, "epsilonStatic": 15.15, "Eg": 0.417, "VBO": -0.59,},
    "AlSb": {"m_e": 0.14, "epsilonStatic": 10.9, "Eg": 2.386, "VBO": -0.41,},
    "InSb": {"m_e": 0.0135, "epsilonStatic": 16.8, "Eg": 0.235, "VBO": 0,},
}

# ALLOY PROPERTIES
alloyproperty = {
    "AlGaAs": {"material1": "AlAs", "material2": "GaAs", "Eg": 0.37,},
    "GaInAs": {"material1": "GaAs", "material2": "InAs", "Eg": 0.415, "m_e": 0.0092},
    "InSbAs": {"material1": "InAs", "material2": "InSb", "Eg": 0.067, "m_e": 0.035,},
    "AlInSb": {"material1": "AlSb", "material2": "InSb", "Eg": 0.43,},
}
