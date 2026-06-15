"""
Extended STO-3G Basis Set -- Full Periodic Table (H through Xe, Z=1-54)
========================================================================
Uses Slater orbital exponents (zeta) and universal STO-3G fitting.

For an STO with principal quantum number n and exponent zeta,
the 3 Gaussian exponents are: alpha_k = alpha_k^(n) * zeta^2

Universal contraction coefficients are the SAME for all elements.

References:
  Hehre, Stewart, Pople, J. Chem. Phys. 51, 2657 (1969)
  Hehre, Ditchfield, Stewart, Pople, J. Chem. Phys. 52, 2769 (1970)
  Pietro et al., JACS 104, 5039 (1982) -- extension to 3rd row+
"""

import numpy as np

# =====================================================================
# UNIVERSAL STO-3G FITTING PARAMETERS (for zeta=1 STO)
# =====================================================================
# alpha_k^(n) for each principal quantum number n
# Contraction coefficients d_k are universal

STO3G_UNIVERSAL = {
    1: {  # 1s shell
        'exponents': [3.42525091, 0.62391373, 0.16885540],
        's_coeffs': [0.1543289673, 0.5353281423, 0.4446345422],
    },
    2: {  # 2sp shell
        'exponents': [2.94124940, 0.68348310, 0.22228990],
        's_coeffs': [-0.0999672292, 0.3995128261, 0.7001154689],
        'p_coeffs': [0.1559162750, 0.6076837186, 0.3919573931],
    },
    3: {  # 3sp shell
        'exponents': [2.21270140, 0.51380660, 0.16714440],
        's_coeffs': [-0.0999672292, 0.3995128261, 0.7001154689],
        'p_coeffs': [0.1559162750, 0.6076837186, 0.3919573931],
    },
    4: {  # 4sp shell
        'exponents': [1.75895000, 0.40885800, 0.13300200],
        's_coeffs': [-0.0999672292, 0.3995128261, 0.7001154689],
        'p_coeffs': [0.1559162750, 0.6076837186, 0.3919573931],
    },
    5: {  # 5sp shell
        'exponents': [1.46100000, 0.33960000, 0.11050000],
        's_coeffs': [-0.0999672292, 0.3995128261, 0.7001154689],
        'p_coeffs': [0.1559162750, 0.6076837186, 0.3919573931],
    },
    '3d': {  # 3d shell
        'exponents': [2.94124940, 0.68348310, 0.22228990],
        'd_coeffs': [0.1559162750, 0.6076837186, 0.3919573931],
    },
    '4d': {  # 4d shell
        'exponents': [2.21270140, 0.51380660, 0.16714440],
        'd_coeffs': [0.1559162750, 0.6076837186, 0.3919573931],
    },
}

# =====================================================================
# SLATER ORBITAL EXPONENTS (zeta) FOR ALL ELEMENTS
# =====================================================================
# Format: {Z: {'1s': zeta, '2sp': zeta, '3sp': zeta, '3d': zeta, ...}}
# Sources: Clementi & Raimondi (1963), Burns (1964), Pietro et al. (1982)

SLATER_ZETAS = {
    # Row 1
    1:  {'1s': 1.24},                                              # H
    2:  {'1s': 1.6875},                                            # He
    # Row 2
    3:  {'1s': 2.6906, '2sp': 0.6396},                            # Li
    4:  {'1s': 3.6848, '2sp': 0.9560},                            # Be
    5:  {'1s': 4.6795, '2sp': 1.2116},                            # B
    6:  {'1s': 5.6727, '2sp': 1.6083},                            # C
    7:  {'1s': 6.6703, '2sp': 1.9237},                            # N
    8:  {'1s': 7.6579, '2sp': 2.2266},                            # O
    9:  {'1s': 8.6501, '2sp': 2.5500},                            # F
    10: {'1s': 9.6421, '2sp': 2.8792},                            # Ne
    # Row 3
    11: {'1s': 10.6259, '2sp': 3.2857, '3sp': 0.7648},            # Na
    12: {'1s': 11.6089, '2sp': 3.6960, '3sp': 1.0180},            # Mg
    13: {'1s': 12.5918, '2sp': 4.0886, '3sp': 1.3552},            # Al
    14: {'1s': 13.5745, '2sp': 4.4913, '3sp': 1.5575},            # Si
    15: {'1s': 14.5578, '2sp': 4.8944, '3sp': 1.8284},            # P
    16: {'1s': 15.5409, '2sp': 5.2998, '3sp': 2.0224},            # S
    17: {'1s': 16.5239, '2sp': 5.7066, '3sp': 2.2547},            # Cl
    18: {'1s': 17.5075, '2sp': 6.1152, '3sp': 2.4888},            # Ar
    # Row 4
    19: {'1s': 18.4895, '2sp': 6.5234, '3sp': 2.2340, '4sp': 0.8738},  # K
    20: {'1s': 19.4730, '2sp': 6.9321, '3sp': 2.4490, '4sp': 1.1074},  # Ca
    21: {'1s': 20.4566, '2sp': 7.3409, '3sp': 2.6640, '3d': 1.3380, '4sp': 0.8170},  # Sc
    22: {'1s': 21.4409, '2sp': 7.7496, '3sp': 2.8060, '3d': 1.8030, '4sp': 0.8270},  # Ti
    23: {'1s': 22.4256, '2sp': 8.1586, '3sp': 2.9650, '3d': 2.1580, '4sp': 0.8380},  # V
    24: {'1s': 23.4104, '2sp': 8.5676, '3sp': 3.1280, '3d': 2.4410, '4sp': 0.8760},  # Cr
    25: {'1s': 24.3957, '2sp': 8.9768, '3sp': 3.2910, '3d': 2.6800, '4sp': 0.8730},  # Mn
    26: {'1s': 25.3810, '2sp': 9.3859, '3sp': 3.4530, '3d': 2.9120, '4sp': 0.8590},  # Fe
    27: {'1s': 26.3668, '2sp': 9.7953, '3sp': 3.6140, '3d': 3.1430, '4sp': 0.8580},  # Co
    28: {'1s': 27.3526, '2sp': 10.2046, '3sp': 3.7750, '3d': 3.3680, '4sp': 0.8570},  # Ni
    29: {'1s': 28.3386, '2sp': 10.6141, '3sp': 3.9360, '3d': 3.5580, '4sp': 0.8990},  # Cu
    30: {'1s': 29.3245, '2sp': 11.0235, '3sp': 4.0960, '3d': 3.7640, '4sp': 1.1070},  # Zn
    31: {'1s': 30.3097, '2sp': 11.4328, '3sp': 4.2110, '3d': 4.4300, '4sp': 1.2930},  # Ga
    32: {'1s': 31.2945, '2sp': 11.8424, '3sp': 4.4690, '3d': 5.1400, '4sp': 1.4770},  # Ge
    33: {'1s': 32.2793, '2sp': 12.2520, '3sp': 4.7280, '3d': 5.8500, '4sp': 1.6640},  # As
    34: {'1s': 33.2642, '2sp': 12.6615, '3sp': 4.9860, '3d': 6.5600, '4sp': 1.8510},  # Se
    35: {'1s': 34.2493, '2sp': 13.0712, '3sp': 5.2450, '3d': 7.2700, '4sp': 2.0360},  # Br
    36: {'1s': 35.2344, '2sp': 13.4808, '3sp': 5.5030, '3d': 7.9800, '4sp': 2.2210},  # Kr
    # Row 5
    37: {'1s': 36.2195, '2sp': 13.8905, '3sp': 5.7630, '3d': 8.6900, '4sp': 2.0750, '5sp': 0.8860},  # Rb
    38: {'1s': 37.2046, '2sp': 14.3001, '3sp': 6.0230, '3d': 9.4000, '4sp': 2.2600, '5sp': 1.1210},  # Sr
    39: {'1s': 38.1897, '2sp': 14.7097, '3sp': 6.2830, '3d': 10.110, '4sp': 2.4440, '4d': 1.3600, '5sp': 0.8520},  # Y
    40: {'1s': 39.1748, '2sp': 15.1194, '3sp': 6.5430, '3d': 10.820, '4sp': 2.5700, '4d': 1.7900, '5sp': 0.8620},  # Zr
    41: {'1s': 40.1599, '2sp': 15.5290, '3sp': 6.8030, '3d': 11.530, '4sp': 2.7140, '4d': 2.1200, '5sp': 0.8670},  # Nb
    42: {'1s': 41.1451, '2sp': 15.9387, '3sp': 7.0630, '3d': 12.240, '4sp': 2.8580, '4d': 2.4200, '5sp': 0.8800},  # Mo
    43: {'1s': 42.1302, '2sp': 16.3483, '3sp': 7.3230, '3d': 12.950, '4sp': 3.0020, '4d': 2.6900, '5sp': 0.8900},  # Tc
    44: {'1s': 43.1153, '2sp': 16.7580, '3sp': 7.5830, '3d': 13.660, '4sp': 3.1460, '4d': 2.9200, '5sp': 0.9000},  # Ru
    45: {'1s': 44.1004, '2sp': 17.1676, '3sp': 7.8430, '3d': 14.370, '4sp': 3.2900, '4d': 3.1300, '5sp': 0.9100},  # Rh
    46: {'1s': 45.0855, '2sp': 17.5773, '3sp': 8.1030, '3d': 15.080, '4sp': 3.4340, '4d': 3.3100, '5sp': 0.9200},  # Pd
    47: {'1s': 46.0706, '2sp': 17.9869, '3sp': 8.3630, '3d': 15.790, '4sp': 3.5780, '4d': 3.5200, '5sp': 1.0200},  # Ag
    48: {'1s': 47.0558, '2sp': 18.3966, '3sp': 8.6230, '3d': 16.500, '4sp': 3.7220, '4d': 3.7100, '5sp': 1.1700},  # Cd
    49: {'1s': 48.0409, '2sp': 18.8062, '3sp': 8.8830, '3d': 17.210, '4sp': 3.8480, '4d': 4.3200, '5sp': 1.3000},  # In
    50: {'1s': 49.0260, '2sp': 19.2159, '3sp': 9.1430, '3d': 17.920, '4sp': 4.0470, '4d': 4.9800, '5sp': 1.4770},  # Sn
    51: {'1s': 50.0111, '2sp': 19.6255, '3sp': 9.4030, '3d': 18.630, '4sp': 4.2460, '4d': 5.6500, '5sp': 1.6580},  # Sb
    52: {'1s': 50.9963, '2sp': 20.0352, '3sp': 9.6630, '3d': 19.340, '4sp': 4.4450, '4d': 6.3100, '5sp': 1.8380},  # Te
    53: {'1s': 51.9814, '2sp': 20.4448, '3sp': 9.9230, '3d': 20.050, '4sp': 4.6440, '4d': 6.9700, '5sp': 2.0180},  # I
    54: {'1s': 52.9665, '2sp': 20.8545, '3sp': 10.183, '3d': 20.760, '4sp': 4.8430, '4d': 7.6300, '5sp': 2.1980},  # Xe
}

# Element symbols indexed by Z
ELEMENT_SYMBOLS = {
    1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
    11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',
    19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',
    27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',
    35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',
    43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',
    51:'Sb',52:'Te',53:'I',54:'Xe',
}
SYMBOL_TO_Z = {v: k for k, v in ELEMENT_SYMBOLS.items()}


def get_sto3g_shells(Z):
    """
    Generate STO-3G shell data for element with atomic number Z.

    Returns list of shell dicts with 'type', 'exponents', 'coefficients'.
    """
    if Z not in SLATER_ZETAS:
        raise ValueError(f"STO-3G not available for Z={Z}. Available: Z=1-54")

    zetas = SLATER_ZETAS[Z]
    shells = []

    # Process shells in order
    shell_order = ['1s', '2sp', '3sp', '3d', '4sp', '4d', '5sp']

    for shell_name in shell_order:
        if shell_name not in zetas:
            continue
        zeta = zetas[shell_name]

        if shell_name == '1s':
            univ = STO3G_UNIVERSAL[1]
            exps = [a * zeta**2 for a in univ['exponents']]
            shells.append({
                'type': '1s',
                'exponents': exps,
                'coefficients': {'s': list(univ['s_coeffs'])},
            })

        elif shell_name.endswith('sp'):
            n = int(shell_name[0])
            univ = STO3G_UNIVERSAL[n]
            exps = [a * zeta**2 for a in univ['exponents']]
            shells.append({
                'type': shell_name,
                'exponents': exps,
                'coefficients': {
                    's': list(univ['s_coeffs']),
                    'p': list(univ['p_coeffs']),
                },
            })

        elif shell_name.endswith('d'):
            n_key = shell_name  # '3d' or '4d'
            univ = STO3G_UNIVERSAL[n_key]
            exps = [a * zeta**2 for a in univ['exponents']]
            shells.append({
                'type': shell_name,
                'exponents': exps,
                'coefficients': {'d': list(univ['d_coeffs'])},
            })

    return shells


def count_basis_functions(Z):
    """Count number of basis functions for element Z in STO-3G."""
    shells = get_sto3g_shells(Z)
    n = 0
    for shell in shells:
        coeffs = shell['coefficients']
        if 's' in coeffs:
            n += 1
        if 'p' in coeffs:
            n += 3
        if 'd' in coeffs:
            n += 6  # 6 Cartesian d functions (or 5 spherical)
    return n
