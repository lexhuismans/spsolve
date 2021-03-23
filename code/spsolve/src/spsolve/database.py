"""
Database
"""
# MATERIAL PROPERTIES
materialproperty = {
    'GaAs':{
        'm_e':.067, # conduction band effective mass (relative to electron mass)
        'm_hh':.45, # heavy hole band effective mass
        'm_lh':.087, # light hole band effective mass 
        'epsilonStatic':12.90, #dielectric constant
        'Eg':1.4223,#1.42 # (ev) band gap
        'Ep':28.8, # (eV) k.p matrix element (used for non-parabolicity calculation (Vurgaftman2001)
        'F':-1.94, # Kane parameter (used for non-parabolicity calculation (Vurgaftman2001)
        'band_offset':0.65, # conduction band/valence band offset ratio for GaAs - AlGaAs heterojunctions
        'm_e_alpha':5.3782e18, # conduction band non-parabolicity variable for linear relation (Nelson approach)
        # Valence band constants 
        'delta':0.28, # (eV) Spin split-off energy gap
        # below used by aestimo_numpy_h
        'GA1':6.8, #luttinger parameter
        'GA2':1.9, #luttinger parameter
        'GA3':2.73, #luttinger parameter
        'C11':11.879, # (GPa) Elastic Constants
        'C12':5.376, # (GPa) Elastic Constants
        'a0':5.6533, # (A)Lattice constant
        'Ac':-7.17, # (eV) deformation potentials (Van de Walle formalism)
        'Av':1.16, # (eV) deformation potentials (Van de Walle formalism)
        'B':-1.7, # (eV) shear deformation potential (Van de Walle formalism)
        'TAUN0':0.1E-7,# Electron SRH life time
        'TAUP0':0.1E-7,# Hole SRH life time
        'mun0':0.1,# Electron Mobility in m2/V-s
        'mup0':0.02,# Electron Mobility in m2/V-s
        'Cn0':2.8e-31,# generation recombination model parameters [cm**6/s]
        'Cp0':2.8e-32,# generation recombination model parameters [cm**6/s]
        'BETAN':2.0,# Parameter in calculatation of the Field Dependant Mobility
        'BETAP':1.0,# Parameter in calculatation of the Field Dependant Mobility
        'VSATN':3e5,# Saturation Velocity of Electrons
        'VSATP':6e5, # Saturation Velocity of Holes
        'AVb_E':-6.92#Average Valence Band Energy or the absolute energy level
    },
    'AlAs':{
        'm_e':.15,
        'm_hh':.51,
        'm_lh':.18,
        'epsilonStatic':10.06,
        'Eg':3.0,#2.980,
        'Ep':21.1,
        'F':-0.48,
        'band_offset':0.53,
        'm_e_alpha':0.0,
        'GA1':3.45,
        'GA2':0.68,
        'GA3':1.29, 
        'C11':11.879,
        'C12':5.376,
        'a0':5.66, 
        'Ac':-5.64,
        'Av':2.47,
        'B':-1.5,
        'delta':0.28,
        'TAUN0':0.1E-6,
        'TAUP0':0.1E-6,
        'mun0':0.15,
        'mup0':0.1,
        'Cn0':2.8e-31,# generation recombination model parameters [cm**6/s]
        'Cp0':2.8e-32,# generation recombination model parameters [cm**6/s]
        'BETAN':2.0,
        'BETAP':1.0,# Parameter in calculatation of the Field Dependant Mobility
        'VSATN':3e5,# Saturation Velocity of Electrons
        'VSATP':6e5, # Saturation Velocity of Holes
        'AVb_E':-7.49#Average Valence Band Energy or the absolute energy level
    }}

# ALLOY PROPERTIES
alloyproperty = {
    'AlGaAs':{
        'bowing_param':0.37,
        'band_offset':0.65,
        'm_e_alpha':5.3782e18,
        'delta_bowing_param':0.0,
        'a0_sub':5.6533,
        'material1':'AlAs',
        'material2':'GaAs',
        'TAUN0':0.1E-6,
        'TAUP0':0.1E-6,
        'mun0':0.15,
        'mup0':0.1,
        'Cn0':2.8e-31,# generation recombination model parameters [cm**6/s]
        'Cp0':2.8e-32,# generation recombination model parameters [cm**6/s]
        'BETAN':2.0,
        'BETAP':1.0,# Parameter in calculatation of the Field Dependant Mobility
        'VSATN':3e5,# Saturation Velocity of Electrons
        'VSATP':6e5 , # Saturation Velocity of Holes
        'AVb_E':-2.1#Average Valence Band Energy or the absolute energy level
    }
}