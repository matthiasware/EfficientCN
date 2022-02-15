import numpy as np
from scipy.stats import pearsonr

EPS = 1e-9


def activation_coupling_corr(C, U):
    C, pr = normalize_couplings(C, eps_rate=0.5)
    C = C.max(axis=-1)
    U = np.linalg.norm(U, axis=-1)
    return pearsonr(C.flatten(), U.flatten())[0]


def normalize_couplings(C, eps_rate=0.5):
    n_samples, n_l, n_h = C.shape
    ef = 1 / n_h + 1 / n_h * eps_rate
    CN = np.maximum(C - ef, 0)
    CN[CN > 0] += ef
    C = CN
    #
    ar = np.mean(np.linalg.norm(C, axis=-1) > 0, axis=0)
    pr = 1 - ar
    return C, pr


def mm_capsules(C):
    return np.max(C, axis=2).mean(axis=0)


def mma_capsules_n(C, pr):
    _, n_l, _ = C.shape
    assert len(pr) == n_l
    mma = C.max(axis=-1).mean(axis=0) / (1 - pr.flatten() + EPS)
    return mma


def mma_layer(C, pr):
    mma = mma_capsules_n(C, pr)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    mma = (ws * mma).sum()
    return mma


def mm_layer(C, pr):
    mm = mm_capsules(C)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    mm = (ws * mm).sum()
    return mm


def ma_couplings_n(C, pr):
    _, n_l, _ = C.shape
    pr = pr.reshape(n_l, 1)
    return C.mean(axis=0) / (1 - pr + EPS)
#


def stda_couplings_n(C, pr):
    _, n_l, _ = C.shape
    pr = pr.reshape(n_l, 1)
    ma = ma_couplings_n(C, pr)
    p1 = ((C - ma)**2).mean(axis=0) / (1 - pr + EPS)
    p2 = ma**2 * pr / (1 - pr + EPS)
    p3 = p1 - p2
    p3 = np.maximum(0, p3)
    return np.sqrt(p3)
#


def lmstda_capsules(C, pr):
    mstda = stda_couplings_n(C, pr).mean(axis=1)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    lmstda = (mstda * ws).sum()
    return lmstda


def lstda_capsules(C, pr):
    stda = stda_couplings_n(C, pr).mean(axis=1)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    lmstda = (mstda * ws).sum()
    return lmstda


def dyc_capsules_n1(C, pr):
    n_samples, n_l, n_h = C.shape
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    masd = stda_couplings_n(C, pr).mean(axis=1)
    dyc = masd / std_pr
    #dyc = dyc.mean() / (1 - pr.mean() + 1e-9)
    return dyc


def dyc_capsules_n2(C, pr):
    n_samples, n_l, n_h = C.shape
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    masd = stda_couplings_n(C, pr).mean(axis=1)
    mx = C.max(axis=(0, 2))
    dyc = masd / (std_pr * mx + EPS)
    return dyc


def dyc_capsules_n3(C, pr):
    n_samples, n_l, n_h = C.shape
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    stda = stda_couplings_n(C, pr)
    mstda = stda.mean(axis=1)


def dycm_capsules(C, pr):
    mstda = stda_couplings_n(C, pr).mean(axis=1)
    mma = mma_capsules_n(C, pr)
    assert np.all(mstda <= mma)
    dycm = mstda / (mma + EPS)
    return dycm


def ldycm_capsules(C, pr):
    dyc = dycm_capsules(C, pr)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    ldyc = (ws * dyc).sum()
    return ldyc


def dycmx_capsules(C, pr):
    mstda = stda_couplings_n(C, pr).mean(axis=1)
    mx = C.max(axis=(0, -1))
    assert np.all(mstda <= mx)
    dycmx = mstda / (mx + EPS)
    return dycmx


def ldycmx_capsules(C, pr):
    dycmx = dycmx_capsules(C, pr)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    ldycmx = (ws * dycmx).sum()
    return ldycmx


def dycpr_capsules(C, pr):
    n_samples, n_l, n_h = C.shape
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    mstda = stda_couplings_n(C, pr).mean(axis=1)
    dycpr = mstda / std_pr
    return dycpr


def ldycpr_capsules(C, pr):
    dycpr = dycpr_capsules(C, pr)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    ldycpr = (ws * dycpr).sum()
    return ldycpr


def dycmpr_capsules(C, pr):
    n_samples, n_l, n_h = C.shape
    mstda = stda_couplings_n(C, pr).mean(axis=1)
    mma = mma_capsules_n(C, pr)
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    assert np.all(mstda <= mma)
    dycmpr = mstda / (mma * std_pr + EPS)
    return dycmpr


def ldycmpr_capsules(C, pr):
    dycmpr = dycmpr_capsules(C, pr)
    ws = (1 - pr) / (1 - pr + EPS).sum()
    ldycmpr = (ws * dycmpr).sum()
    return ldycmpr


def dycmxpr_capsules(C, pr):
    n_samples, n_l, n_h = C.shape
    mstda = stda_couplings_n(C, pr).mean(axis=1)
    cmx = C.max(axis=(0, -1))
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    assert np.all(mstda <= cmx)
    dycmxpr = mstda / (cmx * std_pr + EPS)
    return dycmxpr


def ldycmxpr_capsules(C, pr):
    n_samples, n_l, n_h = C.shape
    mstda = stda_couplings_n(C, pr).mean(axis=1)
    cmx = C.max(axis=(0, -1))
    std_pr = np.sqrt(1 / n_h * (1 - 1 / n_h))
    assert np.all(mstda <= cmx)
    dycmxpr = mstda / (cmx + EPS)
    #
    ws = (1 - pr) / (1 - pr + EPS).sum()
    ldycmxpr = (ws * dycmxpr).sum() / std_pr
    return ldycmxpr


def rate_dead_capsules_norm(U, eps_mean=1e-2, eps_std=1e-2):
    U_norm = np.linalg.norm(U, axis=2)
    U_norm_mu = U_norm.mean(axis=0)
    U_norm_sd = U_norm.std(axis=0)
    #
    U_dead = (U_norm_sd < eps_mean) * (U_norm_mu < eps_std)
    return U_dead


def rate_inactive_capsules(C, eps_rate=0.5):
    Cn, pr = normalize_couplings(C, eps_rate=eps_rate)
    return pr


def rates_activities(U, C):
    """
        ar_u: rate of not dead capsules
        ar_c: rate of inactive capsules
        ar_uc: rate of inactive not dead capsules
        ar_ucu: ar_u *  ar_uc, should match as_c
    """
    pr_u = rate_dead_capsules_norm(U)
    ar_u = 1 - pr_u

    pr_c = rate_inactive_capsules(C)
    ar_c = 1 - pr_c

    pr_uc = rate_inactive_capsules(C[:, ar_u > 0, :])
    ar_uc = 1 - pr_uc

    ar_ucu = ar_uc.mean() * ar_u.mean()

    ar_u = ar_u.mean()
    ar_c = ar_c.mean()
    ar_uc = ar_uc.mean()

    assert np.isclose(ar_c, ar_ucu)

    return ar_u, ar_c, ar_uc


def get_vibrance(U, C):
    return rates_activities(U, C)


def get_bonding(C):
    Cn, pr = normalize_couplings(C, eps_rate=0.5)
    lmma = mma_layer(Cn, pr)
    return lmma


def get_dynamics(C):
    C, pr = normalize_couplings(C, eps_rate=0.5)
    ldycmxpr = ldycmxpr_capsules(C, pr)
    return ldycmxpr
