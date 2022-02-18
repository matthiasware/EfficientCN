import numpy as np
from scipy.stats import pearsonr

EPS = 1e-9


# ###########################################
# DESCRIBED IN NOTES
# ###########################################

def rate_dead_capsules_norm(U, eps_mean=1e-2, eps_std=1e-2):
    """"
        In:  U       (n, d) ... Capsules
        Out: U_dead  (n,)   ... Boolean array

        Returns boolean array,
        where each entry indicates if capsule is dead or not.
        A capsule is dead if its mean and std are sufficiently small.
    """
    U_norm = np.linalg.norm(U, axis=2)
    U_norm_mu = U_norm.mean(axis=0)
    U_norm_sd = U_norm.std(axis=0)
    #
    U_dead = (U_norm_sd < eps_mean) * (U_norm_mu < eps_std)
    return U_dead


def rate_inactive_capsules(C, eps_rate=0.5):
    """
        In:
            C (l,h)     Coupling coefficients
            eps_rate    Normalization tolerance
        Out:
            pr (l,)     Rate of inactiveness
                        For each lower lever capsule (alpha)

    """
    Cn, pr = normalize_couplings(C, eps_rate=eps_rate)
    return pr


def normalize_couplings(C, eps_rate=0.5):
    """
        In:
            C  (l,h)  ... Coupling coefficients
            eps_rate  ... Normalization tolerance
        Out:
            Cn (l, h) ... Normalized C
            pr (l)    ... Inactiveness Rate for each capsule l (alpha)
    """
    n_samples, n_l, n_h = C.shape
    ef = 1 / n_h + 1 / n_h * eps_rate
    CN = np.maximum(C - ef, 0)
    CN[CN > 0] += ef
    C = CN
    #
    ar = np.mean(np.linalg.norm(C, axis=-1) > 0, axis=0)
    pr = 1 - ar
    return C, pr


def mm_couplings(C):
    """
        In:  C  (b,l,h) ... Coupling coefficients
        Out: mm (l,)    ... Mean Max Coupling for each capsule l
    """
    return np.max(C, axis=2).mean(axis=0)


def mma_couplings(C, pr):
    """
        In:  C  (b,l,h) ... Coupling coefficients
             pr (l,)    ... Inactiveness Rate (alpha)
        Out: mm (l,)    ... Mean Max Coupling for each capsule l
                            Adjusted for inactive capsules
    """
    _, n_l, _ = C.shape
    assert len(pr) == n_l
    mma = C.max(axis=-1).mean(axis=0) / (1 - pr.flatten() + EPS)
    return mma


def lmma_couplings(C, pr):
    """"
        In:  See mma
        Out: single mma value for the whole layer

    """
    mma = mma_couplings(C, pr)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    mma = (ws * mma).sum()
    return mma


def lmm_couplings(C, pr):
    """"
        In:  See mm
        Out: single mm value for the whole layer

    """
    mm = mm_couplings(C)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    mm = (ws * mm).sum()
    return mm


def ma_couplings(C, pr):
    """
        In:  C  (b,l,h) ... Coupling coefficients
             pr (l,)    ... Inactiveness Rate (alpha)
        Out: m  (l,h)   ... Mean coupling
                             adjusted for inactiveness
    """
    _, n_l, _ = C.shape
    pr = pr.reshape(n_l, 1)
    return C.mean(axis=0) / (1 - pr + EPS)


def stda_couplings(C, pr):
    """
        In:  C  (b,l,h) ... Coupling coefficients
             pr (l,)    ... Inactiveness Rate (alpha)
        Out: m  (l,h)   ... Standard deviation of couplings
                            adjusted for inactiveness
    """
    _, n_l, _ = C.shape
    pr = pr.reshape(n_l, 1)
    ma = ma_couplings(C, pr)
    p1 = ((C - ma)**2).mean(axis=0) / (1 - pr + EPS)
    p2 = ma**2 * pr / (1 - pr + EPS)
    p3 = p1 - p2
    p3 = np.maximum(0, p3)
    return np.sqrt(p3)


def lmstda_couplings(C, pr):
    """
        In: See stda
        Out: single stda value for whole layer
             adjusted for inactiveness
    """
    mstda = stda_couplings(C, pr).mean(axis=1)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    lmstda = (mstda * ws).sum()
    return lmstda


def dycmxpr_couplings(C, pr):
    n_samples, n_l, n_h = C.shape
    mstda = stda_couplings(C, pr).mean(axis=1)
    cmx = C.max(axis=(0, -1))
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    assert np.all(mstda <= cmx)
    dycmxpr = mstda / (cmx * std_pr + EPS)
    return dycmxpr


def ldycmxpr_couplings(C, pr):
    """
        In:  See stda
        Out: dynamics coefficient for whole layer
    """
    n_samples, n_l, n_h = C.shape
    mstda = stda_couplings(C, pr).mean(axis=1)
    cmx = C.max(axis=(0, -1))
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    assert np.all(mstda <= cmx)
    dycmxpr = mstda / (cmx + EPS)
    #
    ws = (1 - pr) / (1 - pr + EPS).sum()
    ldycmxpr = (ws * dycmxpr).sum() / std_pr
    return ldycmxpr


def rates_activities(U, C):
    """
        IN: ...
        OUT: (for layer)
            rnd:   rate of not dead (= alive) capsules
            rac:   rate of active capsules
            racnd: rate of active of not dead capsules
    """
    rnd = 1 - rate_dead_capsules_norm(U)

    rac = 1 - rate_inactive_capsules(C)

    racnd = 1 - rate_inactive_capsules(C[:, rnd > 0, :])

    rnd = rnd.mean()
    rac = rac.mean()
    racnd = racnd.mean()

    assert np.isclose(rnd * racnd, rac)

    return rnd, rac, racnd


def get_vibrance(U, C):
    """
        Vibrance = activity rates
    """
    return rates_activities(U, C)


def get_bonding(C):
    """
        Bonding strength = lmma
    """
    Cn, pr = normalize_couplings(C, eps_rate=0.5)
    lmma = lmma_couplings(Cn, pr)
    return lmma


def get_dynamics(C):
    """
        Dynamics = ldycmxpr
    """
    C, pr = normalize_couplings(C, eps_rate=0.5)
    ldycmxpr = ldycmxpr_couplings(C, pr)
    return ldycmxpr


def activation_coupling_corr(C, U):
    C, pr = normalize_couplings(C, eps_rate=0.5)
    C = C.max(axis=-1)
    U = np.linalg.norm(U, axis=-1)
    return pearsonr(C.flatten(), U.flatten())[0]

# ###########################################
# CURRENTLY UNUSED BUT MAYBE USEFULL
# ###########################################


# def activation_coupling_corr(C, U):
#     C, pr = normalize_couplings(C, eps_rate=0.5)
#     C = C.max(axis=-1)
#     U = np.linalg.norm(U, axis=-1)
#     return pearsonr(C.flatten(), U.flatten())[0]


# def dycm_capsules(C, pr):
#     mstda = stda_couplings(C, pr).mean(axis=1)
#     mma = mma_couplings(C, pr)
#     assert np.all(mstda <= mma)
#     dycm = mstda / (mma + EPS)
#     return dycm


# def ldycm_capsules(C, pr):
#     dyc = dycm_capsules(C, pr)
#     ws = (1 - pr) / (1 - pr + EPS).sum()
#     ldyc = (ws * dyc).sum()
#     return ldyc


# def dycmx_capsules(C, pr):
#     mstda = stda_couplings(C, pr).mean(axis=1)
#     mx = C.max(axis=(0, -1))
#     assert np.all(mstda <= mx)
#     dycmx = mstda / (mx + EPS)
#     return dycmx


# def ldycmx_capsules(C, pr):
#     dycmx = dycmx_capsules(C, pr)
#     ws = (1 - pr) / (1 - pr + EPS).sum()
#     ldycmx = (ws * dycmx).sum()
#     return ldycmx


# def dycpr_capsules(C, pr):
#     n_samples, n_l, n_h = C.shape
#     std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
#     mstda = stda_couplings(C, pr).mean(axis=1)
#     dycpr = mstda / std_pr
#     return dycpr


# def ldycpr_capsules(C, pr):
#     dycpr = dycpr_capsules(C, pr)
#     ws = (1 - pr) / (1 - pr + EPS).sum()
#     ldycpr = (ws * dycpr).sum()
#     return ldycpr


# def dycmpr_capsules(C, pr):
#     n_samples, n_l, n_h = C.shape
#     mstda = stda_couplings(C, pr).mean(axis=1)
#     mma = mma_couplings(C, pr)
#     std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
#     assert np.all(mstda <= mma)
#     dycmpr = mstda / (mma * std_pr + EPS)
#     return dycmpr


# def ldycmpr_capsules(C, pr):
#     dycmpr = dycmpr_capsules(C, pr)
#     ws = (1 - pr) / (1 - pr + EPS).sum()
#     ldycmpr = (ws * dycmpr).sum()
#     return ldycmpr