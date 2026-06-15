"""
Analytical Gaussian Integrals -- McMurchie-Davidson + Obara-Saika
==================================================================
Implements all one- and two-electron integrals over Cartesian GTOs.

Key equations:
  Overlap:  S_ij = <i|j>
  Kinetic:  T_ij = <i|-0.5 nabla^2|j>
  Nuclear:  V_ij = <i|-Z/|r-R_C||j>
  ERI:      (ij|kl) = <ij|1/r_12|kl>

Uses Boys function F_n(x) = int_0^1 t^(2n) exp(-xt^2) dt
and Gaussian Product Theorem throughout.

SIGN CONVENTIONS (critical for correctness):
  AB = A - B  (difference of centers)
  P  = (alpha_a * A + alpha_b * B) / p   (product center)
  XPA = P_x - A_x = alpha_b * (B_x - A_x) / p = -alpha_b * AB_x / p
  XPB = P_x - B_x = alpha_a * (A_x - B_x) / p =  alpha_a * AB_x / p
"""

import numpy as np
from scipy.special import hyp1f1
from .basis import BasisSet, ContractedGaussian, PrimitiveGaussian


# =====================================================================
# BOYS FUNCTION
# =====================================================================

def boys_function(n, x):
    """
    Boys function F_n(x) = int_0^1 t^(2n) exp(-x t^2) dt

    Using confluent hypergeometric (Kummer) function:
      F_n(x) = (1/(2n+1)) * 1F1(n+1/2; n+3/2; -x)

    Limits:
      F_n(0) = 1/(2n+1)
      F_n(large x) ~ (2n-1)!! * sqrt(pi) / (2^(n+1) * x^(n+0.5))
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-15
    big = ~small
    result[small] = 1.0 / (2.0 * n + 1.0)
    if np.any(big):
        result[big] = hyp1f1(n + 0.5, n + 1.5, -x[big]) / (2.0 * n + 1.0)
    if scalar:
        return float(result[0])
    return result


def gaussian_product_center(alpha_a, center_a, alpha_b, center_b):
    """
    Gaussian Product Theorem:
      exp(-a|r-A|^2) * exp(-b|r-B|^2) = K * exp(-(a+b)|r-P|^2)
    where P = (a*A + b*B)/(a+b), K = exp(-a*b/(a+b) * |A-B|^2)
    """
    return (alpha_a * center_a + alpha_b * center_b) / (alpha_a + alpha_b)


# =====================================================================
# HERMITE EXPANSION COEFFICIENTS (McMurchie-Davidson)
# =====================================================================

def _E_table(la, lb, AB_comp, alpha_a, alpha_b):
    """
    Compute Hermite expansion coefficients E^{la,lb}_t using iterative
    McMurchie-Davidson recursion.

    The product of two Cartesian Gaussians is expanded in Hermite Gaussians:
      x_A^i * exp(-a*(x-Ax)^2) * x_B^j * exp(-b*(x-Bx)^2)
        = sum_t E^{ij}_t * H_t(x - Px) * exp(-p*(x-Px)^2)

    Recursion (incrementing i):
      E^{i+1,j}_t = (1/(2p)) E^{ij}_{t-1} + XPA * E^{ij}_t + (t+1) E^{ij}_{t+1}

    Recursion (incrementing j):
      E^{i,j+1}_t = (1/(2p)) E^{ij}_{t-1} + XPB * E^{ij}_t + (t+1) E^{ij}_{t+1}

    Base: E^{00}_0 = exp(-mu * AB^2), E^{00}_{t!=0} = 0

    Args:
        la, lb: angular momentum on centers A, B (for one Cartesian component)
        AB_comp: A_x - B_x (or y, z component)
        alpha_a, alpha_b: Gaussian exponents

    Returns:
        list of E^{la,lb}_t for t = 0, 1, ..., la+lb
    """
    p = alpha_a + alpha_b
    mu = alpha_a * alpha_b / p

    # CRITICAL SIGN CONVENTION:
    # P = (a*A + b*B)/p
    # XPA = P - A = b*(B - A)/p = -b*AB/p   where AB = A - B
    # XPB = P - B = a*(A - B)/p =  a*AB/p
    XPA = -alpha_b * AB_comp / p
    XPB =  alpha_a * AB_comp / p

    max_t = la + lb

    # E3[i][j] is a list of length (i+j+1) holding E^{ij}_t
    # We only need current and previous columns, but for clarity store all
    # Use a dict of dicts for safe access
    E = {}
    for i in range(la + 1):
        for j in range(lb + 1):
            E[(i, j)] = [0.0] * (i + j + 1)

    # Base case
    E[(0, 0)][0] = np.exp(-mu * AB_comp**2)

    # Build up i (j=0) using XPA
    for i in range(la):
        for t in range(i + 2):  # t ranges 0..i+1
            val = XPA * (E[(i, 0)][t] if t <= i else 0.0)
            if t > 0:
                val += (1.0 / (2.0 * p)) * (E[(i, 0)][t - 1] if t - 1 <= i else 0.0)
            if t + 1 <= i:
                val += (t + 1) * E[(i, 0)][t + 1]
            E[(i + 1, 0)][t] = val

    # Build up j for each i, using XPB
    for i in range(la + 1):
        for j in range(lb):
            for t in range(i + j + 2):  # t ranges 0..i+j+1
                val = XPB * (E[(i, j)][t] if t <= i + j else 0.0)
                if t > 0:
                    val += (1.0 / (2.0 * p)) * (E[(i, j)][t - 1] if t - 1 <= i + j else 0.0)
                if t + 1 <= i + j:
                    val += (t + 1) * E[(i, j)][t + 1]
                E[(i, j + 1)][t] = val

    return E[(la, lb)]


# =====================================================================
# OVERLAP INTEGRALS (Obara-Saika)
# =====================================================================

def _overlap_1d(l1, l2, PA, PB, gamma):
    """
    1D overlap via Obara-Saika recursion.
    S^{i,j}: build i first (j=0), then j.

    S^{i+1,0} = PA * S^{i,0} + i/(2p) * S^{i-1,0}
    S^{i,j+1} = PB * S^{i,j} + j/(2p) * S^{i,j-1} + i/(2p) * S^{i-1,j}

    Base: S^{0,0} = 1 (the exponential prefactor is handled outside)
    """
    val = np.zeros((l1 + 2, l2 + 2))  # extra padding to avoid bounds checks
    val[0, 0] = 1.0
    for i in range(1, l1 + 1):
        val[i, 0] = PA * val[i - 1, 0]
        if i >= 2:
            val[i, 0] += (i - 1) * val[i - 2, 0] / (2.0 * gamma)
    for j in range(1, l2 + 1):
        for i in range(l1 + 1):
            val[i, j] = PB * val[i, j - 1]
            if j >= 2:
                val[i, j] += (j - 1) * val[i, j - 2] / (2.0 * gamma)
            if i >= 1:
                val[i, j] += i * val[i - 1, j - 1] / (2.0 * gamma)
    return val[l1, l2]


def overlap_primitive(a, b):
    """Overlap integral <a|b> between two primitive Gaussians."""
    gamma = a.alpha + b.alpha
    mu = a.alpha * b.alpha / gamma
    AB2 = np.sum((a.center - b.center) ** 2)
    P = gaussian_product_center(a.alpha, a.center, b.alpha, b.center)
    PA = P - a.center
    PB = P - b.center
    pre = np.exp(-mu * AB2) * (np.pi / gamma) ** 1.5
    sx = _overlap_1d(a.l, b.l, PA[0], PB[0], gamma)
    sy = _overlap_1d(a.m, b.m, PA[1], PB[1], gamma)
    sz = _overlap_1d(a.n, b.n, PA[2], PB[2], gamma)
    return a.norm * b.norm * a.coeff * b.coeff * pre * sx * sy * sz


def overlap_contracted(a, b):
    """Overlap between contracted Gaussians."""
    s = 0.0
    for pa in a.primitives:
        for pb in b.primitives:
            s += overlap_primitive(pa, pb)
    return s


# =====================================================================
# KINETIC ENERGY INTEGRALS
# =====================================================================

def kinetic_primitive(a, b):
    """
    Kinetic energy integral T_ab = <a|-0.5 nabla^2|b>

    For each Cartesian direction d:
      T_d = -0.5 * [l_b(l_b-1)*S(l_a, l_b-2) - 2*alpha_b*(2l_b+1)*S(l_a, l_b)
             + 4*alpha_b^2 * S(l_a, l_b+2)]
    T_ab = T_x*S_y*S_z + S_x*T_y*S_z + S_x*S_y*T_z
    """
    gamma = a.alpha + b.alpha
    mu = a.alpha * b.alpha / gamma
    AB2 = np.sum((a.center - b.center) ** 2)
    P = gaussian_product_center(a.alpha, a.center, b.alpha, b.center)
    PA = P - a.center
    PB = P - b.center
    pre = np.exp(-mu * AB2) * (np.pi / gamma) ** 1.5

    def S(la, lb, PAd, PBd):
        if la < 0 or lb < 0:
            return 0.0
        return _overlap_1d(la, lb, PAd, PBd, gamma)

    def T_comp(la, lb, PAd, PBd):
        t = 0.0
        if lb >= 2:
            t += -0.5 * lb * (lb - 1) * S(la, lb - 2, PAd, PBd)
        t += b.alpha * (2 * lb + 1) * S(la, lb, PAd, PBd)
        t -= 2.0 * b.alpha ** 2 * S(la, lb + 2, PAd, PBd)
        return t

    sx = S(a.l, b.l, PA[0], PB[0])
    sy = S(a.m, b.m, PA[1], PB[1])
    sz = S(a.n, b.n, PA[2], PB[2])
    tx = T_comp(a.l, b.l, PA[0], PB[0])
    ty = T_comp(a.m, b.m, PA[1], PB[1])
    tz = T_comp(a.n, b.n, PA[2], PB[2])

    val = tx * sy * sz + sx * ty * sz + sx * sy * tz
    return a.norm * b.norm * a.coeff * b.coeff * pre * val


def kinetic_contracted(a, b):
    """Kinetic energy between contracted Gaussians."""
    t = 0.0
    for pa in a.primitives:
        for pb in b.primitives:
            t += kinetic_primitive(pa, pb)
    return t


# =====================================================================
# HERMITE COULOMB INTEGRALS (for nuclear attraction and ERI)
# =====================================================================

def _hermite_coulomb(t, u, v, n, p, RPC):
    """
    Hermite Coulomb integral R^n_{tuv}(p, RPC) via recursion.

    Base: R^n_{0,0,0} = (-2p)^n * F_n(p * |RPC|^2)

    Recursion:
      R^n_{t+1,u,v} = t * R^{n+1}_{t-1,u,v} + RPC_x * R^{n+1}_{t,u,v}
    (similarly for u with RPC_y, v with RPC_z)
    """
    if t < 0 or u < 0 or v < 0:
        return 0.0
    RPC2 = np.dot(RPC, RPC)
    if t == u == v == 0:
        return (-2.0 * p) ** n * boys_function(n, p * RPC2)
    elif t >= 1:
        return ((t - 1) * _hermite_coulomb(t - 2, u, v, n + 1, p, RPC) +
                RPC[0] * _hermite_coulomb(t - 1, u, v, n + 1, p, RPC))
    elif u >= 1:
        return ((u - 1) * _hermite_coulomb(t, u - 2, v, n + 1, p, RPC) +
                RPC[1] * _hermite_coulomb(t, u - 1, v, n + 1, p, RPC))
    else:  # v >= 1
        return ((v - 1) * _hermite_coulomb(t, u, v - 2, n + 1, p, RPC) +
                RPC[2] * _hermite_coulomb(t, u, v - 1, n + 1, p, RPC))


# =====================================================================
# NUCLEAR ATTRACTION INTEGRALS
# =====================================================================

def nuclear_primitive(a, b, C, Z):
    """
    Nuclear attraction integral: V_ab = -Z * <a|1/|r-C||b>

    McMurchie-Davidson:
      V_ab = -Z * (2*pi/p) * sum_{tuv} E^x_t * E^y_u * E^z_v * R^0_{tuv}(p, P-C)
    """
    p = a.alpha + b.alpha
    P = gaussian_product_center(a.alpha, a.center, b.alpha, b.center)
    RPC = P - C
    AB = a.center - b.center

    # Compute E coefficients for each Cartesian direction
    Ex = _E_table(a.l, b.l, AB[0], a.alpha, b.alpha)
    Ey = _E_table(a.m, b.m, AB[1], a.alpha, b.alpha)
    Ez = _E_table(a.n, b.n, AB[2], a.alpha, b.alpha)

    val = 0.0
    for t in range(a.l + b.l + 1):
        for u in range(a.m + b.m + 1):
            for v in range(a.n + b.n + 1):
                val += Ex[t] * Ey[u] * Ez[v] * _hermite_coulomb(t, u, v, 0, p, RPC)

    return -Z * a.norm * b.norm * a.coeff * b.coeff * (2.0 * np.pi / p) * val


def nuclear_contracted(a, b, C, Z):
    """Nuclear attraction for contracted Gaussians."""
    v = 0.0
    for pa in a.primitives:
        for pb in b.primitives:
            v += nuclear_primitive(pa, pb, C, Z)
    return v


# =====================================================================
# TWO-ELECTRON REPULSION INTEGRALS (ERIs)
# =====================================================================

def eri_primitive(a, b, c, d):
    """
    Electron repulsion integral: (ab|cd)

    McMurchie-Davidson:
      (ab|cd) = (2*pi^2.5) / (p*q*sqrt(p+q))
                * sum_{tuv,tau_nu_phi} E^{ab}_{tuv} * E^{cd}_{tau,nu,phi}
                * (-1)^(tau+nu+phi) * R_{t+tau,u+nu,v+phi}(alpha, P-Q)

    where p = a_a + a_b, q = a_c + a_d, alpha = p*q/(p+q)
    """
    p = a.alpha + b.alpha
    q = c.alpha + d.alpha
    alpha_pq = p * q / (p + q)

    P = gaussian_product_center(a.alpha, a.center, b.alpha, b.center)
    Q = gaussian_product_center(c.alpha, c.center, d.alpha, d.center)
    RPQ = P - Q
    AB = a.center - b.center
    CD = c.center - d.center

    # Compute E coefficients
    E1x = _E_table(a.l, b.l, AB[0], a.alpha, b.alpha)
    E1y = _E_table(a.m, b.m, AB[1], a.alpha, b.alpha)
    E1z = _E_table(a.n, b.n, AB[2], a.alpha, b.alpha)
    E2x = _E_table(c.l, d.l, CD[0], c.alpha, d.alpha)
    E2y = _E_table(c.m, d.m, CD[1], c.alpha, d.alpha)
    E2z = _E_table(c.n, d.n, CD[2], c.alpha, d.alpha)

    val = 0.0
    for t in range(a.l + b.l + 1):
        for u in range(a.m + b.m + 1):
            for v in range(a.n + b.n + 1):
                for tau in range(c.l + d.l + 1):
                    for nu in range(c.m + d.m + 1):
                        for phi in range(c.n + d.n + 1):
                            sign = (-1) ** (tau + nu + phi)
                            R = _hermite_coulomb(
                                t + tau, u + nu, v + phi, 0, alpha_pq, RPQ)
                            val += (E1x[t] * E1y[u] * E1z[v] *
                                    E2x[tau] * E2y[nu] * E2z[phi] * sign * R)

    prefactor = (2.0 * np.pi ** 2.5) / (p * q * np.sqrt(p + q))
    norms = a.norm * b.norm * c.norm * d.norm
    coeffs = a.coeff * b.coeff * c.coeff * d.coeff
    return norms * coeffs * prefactor * val


def eri_contracted(a, b, c, d):
    """ERI for contracted Gaussians."""
    val = 0.0
    for pa in a.primitives:
        for pb in b.primitives:
            for pc in c.primitives:
                for pd in d.primitives:
                    val += eri_primitive(pa, pb, pc, pd)
    return val


# =====================================================================
# MATRIX CONSTRUCTION
# =====================================================================

def compute_overlap_matrix(basis):
    """Build overlap matrix S. Shape: (N, N)."""
    n = basis.nbasis
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S[i, j] = overlap_contracted(basis[i], basis[j])
            S[j, i] = S[i, j]
    return S


def compute_kinetic_matrix(basis):
    """Build kinetic energy matrix T."""
    n = basis.nbasis
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            T[i, j] = kinetic_contracted(basis[i], basis[j])
            T[j, i] = T[i, j]
    return T


def compute_nuclear_matrix(basis, molecule):
    """Build nuclear attraction matrix V = sum_A -Z_A <mu|1/|r-R_A||nu>."""
    n = basis.nbasis
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v_ij = 0.0
            for atom in molecule.atoms:
                v_ij += nuclear_contracted(basis[i], basis[j], atom.coords, atom.Z)
            V[i, j] = v_ij
            V[j, i] = v_ij
    return V


def compute_eri_tensor(basis):
    """
    Build 4-index ERI tensor: (mu nu|la si).
    Shape: (N, N, N, N). Uses 8-fold permutation symmetry.
    """
    n = basis.nbasis
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(i, n):
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                for l in range(k, n):
                    kl = k * (k + 1) // 2 + l
                    if ij >= kl:
                        val = eri_contracted(basis[i], basis[j], basis[k], basis[l])
                        eri[i, j, k, l] = val
                        eri[j, i, k, l] = val
                        eri[i, j, l, k] = val
                        eri[j, i, l, k] = val
                        eri[k, l, i, j] = val
                        eri[l, k, i, j] = val
                        eri[k, l, j, i] = val
                        eri[l, k, j, i] = val
    return eri


def compute_core_hamiltonian(basis, molecule):
    """H_core = T + V (one-electron part of Hamiltonian)."""
    T = compute_kinetic_matrix(basis)
    V = compute_nuclear_matrix(basis, molecule)
    return T + V, T, V


# =====================================================================
# VALIDATION UTILITIES
# =====================================================================

def validate_overlap_matrix(S):
    """Check overlap matrix properties: symmetric, positive-definite, diag~1."""
    n = S.shape[0]
    issues = []
    # Symmetry
    asym = np.max(np.abs(S - S.T))
    if asym > 1e-12:
        issues.append(f"S not symmetric: max asymmetry = {asym:.2e}")
    # Positive-definite
    eigvals = np.linalg.eigvalsh(S)
    if np.any(eigvals < -1e-10):
        issues.append(f"S not positive-definite: min eigenvalue = {eigvals[0]:.2e}")
    # Diagonal elements
    for i in range(n):
        if abs(S[i, i] - 1.0) > 0.5:
            issues.append(f"S[{i},{i}] = {S[i,i]:.6f} (expected ~1.0)")
    return issues


def validate_eri_symmetry(eri, tol=1e-10):
    """Check 8-fold ERI symmetry."""
    n = eri.shape[0]
    max_err = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    ref = eri[i, j, k, l]
                    perms = [eri[j,i,k,l], eri[i,j,l,k], eri[j,i,l,k],
                             eri[k,l,i,j], eri[l,k,i,j], eri[k,l,j,i], eri[l,k,j,i]]
                    for p in perms:
                        max_err = max(max_err, abs(ref - p))
    return max_err < tol, max_err
