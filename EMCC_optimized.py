import math
import os
import sys
import time
import numpy as np
import scipy.optimize  # 新增：引入 Scipy 优化库
from numba import njit
from concurrent.futures import ProcessPoolExecutor


# ==========================================
#  Numba JIT Compiled Kernels (Physics Engine)
# ==========================================

@njit(fastmath=True)
def calc_R_scalar(dis, Beta1, Beta2, Theta1, Theta2, RP, RT, yy20f):
    """Scalar version for single point calculations inside integrals."""
    term1 = 1.0 + Beta1 * yy20f * (3.0 * math.cos(Theta1) ** 2 - 1.0)
    term2 = 1.0 + Beta2 * yy20f * (3.0 * math.cos(Theta2) ** 2 - 1.0)
    return RP * term1 + RT * term2 + dis


@njit(fastmath=True)
def calc_R_batch(dis_arr, Beta1, Beta2, Theta1, Theta2, RP, RT, yy20f):
    """Vectorized version for arrays of distances."""
    term1 = 1.0 + Beta1 * yy20f * (3.0 * math.cos(Theta1) ** 2 - 1.0)
    term2 = 1.0 + Beta2 * yy20f * (3.0 * math.cos(Theta2) ** 2 - 1.0)
    # numpy array broadcasting
    return RP * term1 + RT * term2 + dis_arr


@njit(fastmath=True)
def funevn_kernel(x, R_dist, Thet1, Thet2, Bet1, Bet2, p):
    """Nuclear Density Integrand."""
    RP, RT = p[4], p[5]
    alph1, alph2 = p[9], p[10]
    rh0 = p[11]
    yy20f = p[12]
    coef1, coef2 = p[14], p[15]

    val_sq = 1.0 - x[0] * x[0]
    si1 = math.sqrt(val_sq if val_sq > 0 else 0.0)

    cos_x2 = math.cos(x[2])
    cos_t1 = math.cos(Thet1)
    sin_t1 = math.sin(Thet1)
    csa1 = si1 * cos_x2 * sin_t1 + x[0] * cos_t1

    dist_sq = x[1] * x[1] + R_dist * R_dist - 2.0 * R_dist * x[1] * x[0]
    denom = math.sqrt(dist_sq if dist_sq > 0 else 1e-10)

    cos_t2 = math.cos(Thet2)
    sin_t2 = math.sin(Thet2)
    term_csa2 = x[1] * (si1 * cos_x2 * sin_t2 + x[0] * cos_t2) - R_dist * cos_t2
    csa2 = term_csa2 / denom

    r1 = coef1 * RP * (1.0 + Bet1 * yy20f * (3.0 * csa1 * csa1 - 1.0))
    r2 = coef2 * RT * (1.0 + Bet2 * yy20f * (3.0 * csa2 * csa2 - 1.0))

    arg1 = (x[1] - r1) / alph1
    arg1 = 50.0 if arg1 > 50.0 else (-50.0 if arg1 < -50.0 else arg1)
    rho1 = rh0 / (1.0 + math.exp(arg1))

    arg2 = (denom - r2) / alph2
    arg2 = 50.0 if arg2 > 50.0 else (-50.0 if arg2 < -50.0 else arg2)
    rho2 = rh0 / (1.0 + math.exp(arg2))

    return rho1 * rho1 * rho2, rho1 * rho2 * rho2, rho1 * rho2


@njit(fastmath=True)
def funevdn_kernel(x, R_dist, Thet1, Thet2, Bet1, Bet2, p):
    """Nuclear Density Derivative Integrand."""
    RP, RT = p[4], p[5]
    alph1, alph2 = p[9], p[10]
    rh0 = p[11]
    yy20f = p[12]
    coef1, coef2 = p[14], p[15]

    val_sq = 1.0 - x[0] * x[0]
    si1 = math.sqrt(val_sq if val_sq > 0 else 0.0)

    cos_x2 = math.cos(x[2])
    cos_t1 = math.cos(Thet1)
    sin_t1 = math.sin(Thet1)
    csa1 = si1 * cos_x2 * sin_t1 + x[0] * cos_t1

    dist_sq = x[1] * x[1] + R_dist * R_dist - 2.0 * R_dist * x[1] * x[0]
    denom = math.sqrt(dist_sq if dist_sq > 0 else 1e-10)

    cos_t2 = math.cos(Thet2)
    sin_t2 = math.sin(Thet2)
    term_csa2 = x[1] * (si1 * cos_x2 * sin_t2 + x[0] * cos_t2) - R_dist * cos_t2
    csa2 = term_csa2 / denom

    r1 = coef1 * RP * (1.0 + Bet1 * yy20f * (3.0 * csa1 * csa1 - 1.0))
    r2 = coef2 * RT * (1.0 + Bet2 * yy20f * (3.0 * csa2 * csa2 - 1.0))

    arg1 = (x[1] - r1) / alph1
    arg1 = 50.0 if arg1 > 50.0 else arg1
    rho1 = rh0 / (1.0 + math.exp(arg1))

    arg2 = (denom - r2) / alph2
    arg2 = 50.0 if arg2 > 50.0 else arg2
    exp_term = math.exp(arg2)
    rho2 = rh0 / (1.0 + exp_term)

    fact = -exp_term / (1.0 + exp_term) / alph2
    term1 = (R_dist - x[1] * x[0]) / denom

    csa2_sq = csa2 * csa2
    term2 = 6.0 * coef2 * yy20f * Bet2 * RT * (
            cos_t2 * csa2 / denom + (R_dist - x[1] * x[0]) * csa2_sq / dist_sq
    )
    drho2 = fact * rho2 * (term1 + term2)

    return rho1 * rho1 * drho2, 2.0 * rho1 * rho2 * drho2, rho1 * drho2


@njit(fastmath=True)
def integration_3d_jit(R_dist, Thet1, Thet2, Bet1, Bet2, p, is_deriv, gl_nodes, gl_weights):
    """
    现代化高斯-勒让德积分实现 (Modernized Gauss-Legendre Integration)
    """
    # J=2: [0, rcut]
    rcut = p[13]
    j2_scale = 0.5 * rcut
    j2_shift = 0.5 * rcut

    # J=3:[0, 2*PI]
    PI = p[17]
    j3_scale = PI
    j3_shift = PI

    total_v1, total_v2, total_v3 = 0.0, 0.0, 0.0
    n_points = len(gl_nodes)

    for i in range(n_points):
        x0 = gl_nodes[i]
        w1 = gl_weights[i]

        sum_j2_v1, sum_j2_v2, sum_j2_v3 = 0.0, 0.0, 0.0

        for j in range(n_points):
            x1 = j2_scale * gl_nodes[j] + j2_shift
            w2 = gl_weights[j] * j2_scale
            jac_r2 = x1 * x1

            sum_j3_v1, sum_j3_v2, sum_j3_v3 = 0.0, 0.0, 0.0

            for k in range(n_points):
                x2 = j3_scale * gl_nodes[k] + j3_shift
                w3 = gl_weights[k] * j3_scale

                x_arr = np.array([x0, x1, x2])

                if is_deriv:
                    v1, v2, v3 = funevdn_kernel(x_arr, R_dist, Thet1, Thet2, Bet1, Bet2, p)
                else:
                    v1, v2, v3 = funevn_kernel(x_arr, R_dist, Thet1, Thet2, Bet1, Bet2, p)

                sum_j3_v1 += v1 * w3
                sum_j3_v2 += v2 * w3
                sum_j3_v3 += v3 * w3

            factor = jac_r2 * w2
            sum_j2_v1 += sum_j3_v1 * factor
            sum_j2_v2 += sum_j3_v2 * factor
            sum_j2_v3 += sum_j3_v3 * factor

        total_v1 += sum_j2_v1 * w1
        total_v2 += sum_j2_v2 * w1
        total_v3 += sum_j2_v3 * w1

    return total_v1, total_v2, total_v3


@njit(fastmath=True)
def volume_jit(ras, betas):
    t = np.array([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459])
    c = np.array([0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851])
    s = 0.0
    dn1, up1 = -1.0, 1.0
    d1 = 0.5 * (up1 - dn1) / 4.0

    for seg1 in range(4):
        cc1 = d1 + dn1 + seg1 * 2.0 * d1
        for k1 in range(5):
            x0 = d1 * t[k1] + cc1
            dn2 = 0.0
            up2 = ras * (1.0 + betas * 0.315391565 * (3.0 * x0 * x0 - 1.0))
            d2 = 0.5 * (up2 - dn2) / 4.0

            inner_s = 0.0
            for seg2 in range(4):
                cc2 = d2 + dn2 + seg2 * 2.0 * d2
                for k2 in range(5):
                    x1 = d2 * t[k2] + cc2
                    inner_s += x1 * x1 * c[k2]
            s += inner_s * d2 * c[k1]

    return s * d1 * 2.0 * 3.141592653


@njit(fastmath=True)
def VN_jit(dis, Beta1, Beta2, Theta1, Theta2, p, gl_nodes, gl_weights):
    R_dist = calc_R_scalar(dis, Beta1, Beta2, Theta1, Theta2, p[4], p[5], p[12])
    p_local = p.copy()
    vol1 = volume_jit(p[4], Beta1)
    vol2 = volume_jit(p[5], Beta2)
    p_local[14] = (4. / 3. * p[17] * p[4] ** 3 / vol1) ** 0.3333
    p_local[15] = (4. / 3. * p[17] * p[5] ** 3 / vol2) ** 0.3333

    C0 = 300.0
    Fin = 0.09 + 0.42 * (p[0] - 2. * p[1]) * (p[2] - 2. * p[3]) / (p[0] * p[2])
    Fex = -2.59 + 0.54 * (p[0] - 2. * p[1]) * (p[2] - 2. * p[3]) / (p[0] * p[2])

    vnu1, vnu2, vnu3 = integration_3d_jit(R_dist, Theta1, Theta2, Beta1, Beta2, p_local, False, gl_nodes, gl_weights)
    return C0 * ((Fin - Fex) / p[11] * (vnu1 + vnu2) + Fex * vnu3)


@njit(fastmath=True)
def DVN_jit(dis, Beta1, Beta2, Theta1, Theta2, p, gl_nodes, gl_weights):
    R_dist = calc_R_scalar(dis, Beta1, Beta2, Theta1, Theta2, p[4], p[5], p[12])
    p_local = p.copy()
    vol1 = volume_jit(p[4], Beta1)
    vol2 = volume_jit(p[5], Beta2)
    p_local[14] = (4. / 3. * p[17] * p[4] ** 3 / vol1) ** 0.3333
    p_local[15] = (4. / 3. * p[17] * p[5] ** 3 / vol2) ** 0.3333

    C0 = 300.0
    Fin = 0.09 + 0.42 * (p[0] - 2. * p[1]) * (p[2] - 2. * p[3]) / (p[0] * p[2])
    Fex = -2.59 + 0.54 * (p[0] - 2. * p[1]) * (p[2] - 2. * p[3]) / (p[0] * p[2])

    vnu1, vnu2, vnu3 = integration_3d_jit(R_dist, Theta1, Theta2, Beta1, Beta2, p_local, True, gl_nodes, gl_weights)
    return C0 * ((Fin - Fex) / p[11] * (vnu1 + vnu2) + Fex * vnu3)


@njit(fastmath=True)
def VC_jit(dis, Beta1, Beta2, Theta1, Theta2, p):
    R = calc_R_scalar(dis, Beta1, Beta2, Theta1, Theta2, p[4], p[5], p[12])
    V1 = p[1] * p[3] / R * p[16] / 137.04

    term1 = Beta1 * (3.0 * math.cos(Theta1) ** 2 - 1.0) / 2.0 * p[4] ** 2
    term2 = Beta2 * (3.0 * math.cos(Theta2) ** 2 - 1.0) / 2.0 * p[5] ** 2
    R1 = term1 + term2

    B1 = (Beta1 * (3.0 * math.cos(Theta1) ** 2 - 1.0) / 2.0) ** 2 * p[4] ** 2
    B2 = (Beta2 * (3.0 * math.cos(Theta2) ** 2 - 1.0) / 2.0) ** 2 * p[5] ** 2
    R2 = B1 + B2

    return V1 * (1.0 + math.sqrt(9.0 / (20. * p[17])) * R1 / R ** 2 + 3.0 / (7. * p[17]) * R2 / R ** 2)


@njit(fastmath=True)
def DVC_jit(dis, Beta1, Beta2, Theta1, Theta2, p):
    R = calc_R_scalar(dis, Beta1, Beta2, Theta1, Theta2, p[4], p[5], p[12])
    DV1 = -p[1] * p[3] / R ** 2 * p[16] / 137.04

    term1 = Beta1 * (3.0 * math.cos(Theta1) ** 2 - 1.0) / 2.0 * p[4] ** 2
    term2 = Beta2 * (3.0 * math.cos(Theta2) ** 2 - 1.0) / 2.0 * p[5] ** 2
    R1 = 3.0 * (term1 + term2)

    B1 = (Beta1 * (3.0 * math.cos(Theta1) ** 2 - 1.0) / 2.0) ** 2 * p[4] ** 2
    B2 = (Beta2 * (3.0 * math.cos(Theta2) ** 2 - 1.0) / 2.0) ** 2 * p[5] ** 2
    R2 = 3.0 * (B1 + B2)

    return DV1 * (1.0 + math.sqrt(9.0 / (20. * p[17])) * R1 / R ** 2 + 3.0 / (7. * p[17]) * R2 / R ** 2)


@njit(fastmath=True)
def VCENT_jit(dis, Beta1, Beta2, Theta1, Theta2, L, p):
    R = calc_R_scalar(dis, Beta1, Beta2, Theta1, Theta2, p[4], p[5], p[12])
    return L * (L + 1.0) * p[16] ** 2 / 2.0 / p[6] / R ** 2


@njit(fastmath=True)
def DVCENT_jit(dis, Beta1, Beta2, Theta1, Theta2, L, p):
    R = calc_R_scalar(dis, Beta1, Beta2, Theta1, Theta2, p[4], p[5], p[12])
    return -2.0 * L * (L + 1.0) * p[16] ** 2 / 2.0 / p[6] / R ** 3


@njit(fastmath=True)
def V_total_jit(dis, Beta1, Beta2, Theta1, Theta2, L, p, gl_nodes, gl_weights):
    return VN_jit(dis, Beta1, Beta2, Theta1, Theta2, p, gl_nodes, gl_weights) + \
        VC_jit(dis, Beta1, Beta2, Theta1, Theta2, p) + \
        VCENT_jit(dis, Beta1, Beta2, Theta1, Theta2, L, p)


@njit(fastmath=True)
def DV_total_jit(dis, Beta1, Beta2, Theta1, Theta2, L, p, gl_nodes, gl_weights):
    return DVN_jit(dis, Beta1, Beta2, Theta1, Theta2, p, gl_nodes, gl_weights) + \
        DVC_jit(dis, Beta1, Beta2, Theta1, Theta2, p) + \
        DVCENT_jit(dis, Beta1, Beta2, Theta1, Theta2, L, p)


@njit(fastmath=True)
def V12_jit(dis, Beta1, Beta2, Theta1, Theta2, Betap, Betat, p, gl_nodes, gl_weights):
    vn = VN_jit(dis, Beta1, Beta2, Theta1, Theta2, p, gl_nodes, gl_weights)
    vc = VC_jit(dis, Beta1, Beta2, Theta1, Theta2, p)
    deform_E = 0.5 * p[7] * (Beta1 - Betap) ** 2 + 0.5 * p[8] * (Beta2 - Betat) ** 2
    return vn + vc + deform_E


# --- Vectorized Kernels for Batch Processing ---

@njit(fastmath=True)
def calculate_potentials_batch(dis_arr, Beta1, Beta2, Theta1, Theta2, Betap, Betat, p, gl_nodes, gl_weights):
    n = len(dis_arr)
    vn_arr = np.empty(n)
    vc_arr = np.empty(n)
    vint_arr = np.empty(n)
    r_arr = np.empty(n)

    for i in range(n):
        d = dis_arr[i]
        vn_arr[i] = VN_jit(d, Beta1, Beta2, Theta1, Theta2, p, gl_nodes, gl_weights)
        vc_arr[i] = VC_jit(d, Beta1, Beta2, Theta1, Theta2, p)
        vint_arr[i] = vn_arr[i] + vc_arr[i] + VCENT_jit(d, Beta1, Beta2, Theta1, Theta2, 0, p)
        r_arr[i] = calc_R_scalar(d, Beta1, Beta2, Theta1, Theta2, p[4], p[5], p[12])

    return vn_arr, vc_arr, vint_arr, r_arr


@njit(fastmath=True)
def trans_vectorized_jit(E, L, V_grid, BDF_weighted, C0_base, C1_factor):
    C0 = C0_base * L * (L + 1.0)
    C1 = C1_factor

    arg = V_grid + C0
    sqrt_arg = np.sqrt(np.maximum(arg, 0.0))
    sqrt_V = np.sqrt(np.maximum(V_grid, 0.0))

    sqrt_E = math.sqrt(max(E, 0.0))

    term = sqrt_V * (sqrt_arg - sqrt_E)
    exponent = C1 * term
    safe_mask = exponent < 700.0

    total_trans = 0.0
    n = len(V_grid)
    for i in range(n):
        if safe_mask[i]:
            denom = 1.0 + math.exp(exponent[i])
            total_trans += BDF_weighted[i] / denom

    return total_trans


# ==========================================
#  Parallel Helper
# ==========================================

def compute_potential_surface_point(args):
    # 【修改】：接收外部传入的 Theta1 和 Theta2，取代硬编码
    I1, Betai, Dbet, Betap, Betat, AP, CT, AT, CP, p_array, gl_nodes, gl_weights, Theta1, Theta2 = args
    Beta = Betai + Dbet * I1
    Beta0 = Beta - Betap - Betat
    B0 = math.sqrt(AP * CT / AT / CP)
    Beta1 = B0 / (1. + B0) * Beta0 + Betap
    Beta2 = 1. / (1. + B0) * Beta0 + Betat
    dis = 5.0

    # NEW: Use Brentq instead of manual loop

    # Define a wrapper function for the derivative (force)
    def deriv_func(d):
        return DV_total_jit(d, Beta1, Beta2, Theta1, Theta2, 0, p_array, gl_nodes, gl_weights)

    # 1. Find bracket (sign change)
    found_bracket = False
    while dis > -2.0:
        u_curr = deriv_func(dis)
        u_next = deriv_func(dis - 1.0)
        if u_curr * u_next < 0:
            found_bracket = True
            break
        dis -= 1.0

    if not found_bracket:
        return (I1, 0.0, Beta, Beta1, Beta2)

    # 2. Precise root finding with Brentq
    try:
        # Ensure the bracket is ordered correctly
        final_dis = scipy.optimize.brentq(deriv_func, dis - 1.0, dis)
    except ValueError:
        # Fallback if bracket was somehow invalid
        final_dis = dis - 0.5

    vs_val = V12_jit(final_dis, Beta1, Beta2, Theta1, Theta2, Betap, Betat, p_array, gl_nodes, gl_weights)
    return (I1, vs_val, Beta, Beta1, Beta2)


# ==========================================
#  Main Class
# ==========================================

class EMCCModel:
    def __init__(self):
        self.p = np.zeros(18, dtype=np.float64)
        self.p[16] = 197.327  # HBAR
        self.p[17] = 3.141592653  # PI
        self.ANMASS = 938.0
        self.RB = 0.0;
        self.VB = 0.0;
        self.CURV = 0.0
        self.RB0 = 0.0;
        self.VB0 = 0.0;
        self.VS = 0.0
        self.BETAS = 0.0;
        self.VBtip = 0.0
        self.BTP = 0.0;
        self.BTT = 0.0
        self.Betap = 0.0;
        self.Betat = 0.0
        self.L = 0

        self.V_grid = None
        self.BDF_weighted = None

        # High-Precision Gauss-Legendre Nodes/Weights (60 points)
        self.gl_nodes, self.gl_weights = np.polynomial.legendre.leggauss(60)
        self.gl_nodes = np.ascontiguousarray(self.gl_nodes)
        self.gl_weights = np.ascontiguousarray(self.gl_weights)

    def parse_input_line(self, file_handle):
        while True:
            line = file_handle.readline()
            if not line: return None
            line = line.strip()
            if not line: continue
            content = line.split('!')[0].strip()
            if not content: continue
            parts = content.replace(',', ' ').split()
            return [float(x) for x in parts]

    def run(self):
        if not os.path.exists('barrier_factor.txt'):
            with open('barrier_factor.txt', 'w') as f: f.write("1.0, 1.0\n")

        try:
            f_in = open('EMCCM.IN', 'r')
            f_bar = open('barrier_factor.txt', 'r')
            f_cross = open('CROSS.DAT', 'w')
            f_parti = open('PARTI.DAT', 'w')
            f_trans = open('TRANS.DAT', 'w')
        except FileNotFoundError as e:
            print(f"Error opening file: {e}")
            return

        vals = self.parse_input_line(f_in)
        self.p[0], self.p[1], self.p[2], self.p[3], self.Betap, self.Betat = vals[:6]
        vals = self.parse_input_line(f_in)
        EMIN, EMAX, ESTP = vals[:3]
        vals = self.parse_input_line(f_in)
        Lmax, IVINT = int(vals[0]), int(vals[1])
        vals = self.parse_input_line(f_in)
        self.p[9], self.p[10] = vals[:2]
        vals = self.parse_input_line(f_bar)
        if vals:
            self.barr1, self.barr2 = vals[:2]
        else:
            self.barr1, self.barr2 = 1.0, 1.0

        print(f"{self.barr1} {self.barr2}")
        print("Qvalue")

        self.p[11] = 0.165
        self.p[12] = math.sqrt(5. / self.p[17]) * 0.25
        self.p[13] = 25.0
        AP, AT = self.p[0], self.p[2]
        self.p[4] = 1.28 * AP ** 0.3333 - 0.76 + 0.8 * AP ** (-0.3333)
        self.p[5] = 1.28 * AT ** 0.3333 - 0.76 + 0.8 * AT ** (-0.3333)
        self.p[6] = AP * AT / (AP + AT) * self.ANMASS
        RP, RT, ZP, ZT, PI = self.p[4], self.p[5], self.p[1], self.p[3], self.p[17]
        self.p[7] = (1.0) * ((4.0) * RP ** 2 * 1.084 - 1.5 * 1.44 * ZP ** 2 / PI / 5.0 / RP)
        self.p[8] = (1.0) * ((4.0) * RT ** 2 * 1.084 - 1.5 * 1.44 * ZT ** 2 / PI / 5.0 / RT)

        if IVINT == 1:
            self.L = 0
            dis_start = 10.0
            Theta1, Theta2 = 0.0, 0.0

            RRP = RP * (1. + self.Betap * self.p[12] * (3. * math.cos(Theta1) ** 2 - 1.))
            RRT = RT * (1. + self.Betat * self.p[12] * (3. * math.cos(Theta2) ** 2 - 1.))
            RCP = RRP * (1. - (1. / RRP ** 2))
            RCT = RRT * (1. - (1. / RRT ** 2))
            R_calc = RCP + RCT + 4.5 - (RCT + RCP) / 6.
            barrier_est = (ZP * ZT / 137.038209) / (1.0 / 197.327053) / R_calc
            print('Coulumb barrier is from HUIZENGA formula')
            print(f"{R_calc} {barrier_est}")

            dis_arr = np.arange(dis_start, -2.0, -0.2)

            vn_arr, vc_arr, vint_arr, r_arr = calculate_potentials_batch(
                dis_arr, self.Betap, self.Betat, Theta1, Theta2, self.Betap, self.Betat, self.p,
                self.gl_nodes, self.gl_weights
            )

            for i in range(len(dis_arr)):
                f_parti.write(f" {r_arr[i]:.4f} {dis_arr[i]:.4f} {vn_arr[i]:.6E} {vc_arr[i]:.6E} {vint_arr[i]:.6E}\n")

            print('Calculation of potential with dynamical deformation')
            Betai, Betaf, Dbet = -1.0, 3.0, 0.05
            K = int((Betaf - Betai) / Dbet) + 1
            args_list =[]
            for I1 in range(K):
                # 【修改】：加入当前的 Theta1, Theta2
                args_list.append((I1, Betai, Dbet, self.Betap, self.Betat,
                                  self.p[0], self.p[8], self.p[2], self.p[7], self.p,
                                  self.gl_nodes, self.gl_weights, Theta1, Theta2))

            print(f"Scanning {K} points using Parallel Processing...")
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(compute_potential_surface_point, args_list))
            results.sort(key=lambda x: x[0])
            VSA, BET = [],[]
            for res in results:
                _, vs_val, beta, _, _ = res
                VSA.append(vs_val)
                BET.append(beta)
                f_trans.write(f" {beta} {vs_val}\n")

            self.VS = 1000.0
            idx = 0
            for i, vt in enumerate(VSA):
                if self.VS > vt and vt != 0.0:
                    self.VS = vt
                    idx = i
            print(f"Optimal beta2 and potential: {BET[idx]}, {self.VS}")

        else:
            num_steps = int(round((EMAX - EMIN) / ESTP)) + 1
            print("Initializing Potential Shape (Numba + Parallel + Brentq)...")
            self.L = 0
            self.pot_shape()
            print(f"Potential Shape Initialized. Vb={self.VB0:.4f}, Vs={self.VS:.4f}")

            print("Precomputing BDF distribution...")
            self.precompute_bdf()

            C0_base = self.p[16] ** 2 / (2. * self.p[6] * self.RB ** 2)
            C1_factor = 4. * self.p[17] / self.CURV

            for i in range(num_steps):
                E = EMIN + i * ESTP
                print(f"e={E:.2f}")
                SIGMA = 0.0

                for l_idx in range(Lmax + 1):
                    self.L = l_idx
                    if i == 0 and self.L == 0:
                        f_cross.write(f"Vb= {self.VB0} Vs= {self.VS}\n")

                    if self.RB <= 0.5 or self.CURV == 0.0:
                        trans_val = 0.0
                    else:
                        trans_val = trans_vectorized_jit(E, self.L, self.V_grid, self.BDF_weighted, C0_base, C1_factor)

                    partial_sig = (2. * self.L + 1.) * trans_val * self.p[17] * self.p[16] ** 2 / 2. / self.p[
                        6] / E * 10.
                    SIGMA += partial_sig

                    if trans_val < 1.e-10: break

                    line_str = "{:8.2f}  {:6.2f}    {:14.7E}    {:14.7E}\n".format(E, float(self.L), trans_val,
                                                                                   partial_sig)
                    f_parti.write(line_str)
                    line_str_trans = "{:8.2f}  {:6.2f}    {:14.7E}\n".format(E, float(self.L), trans_val)
                    f_trans.write(line_str_trans)

                f_cross.write(f" {E} {SIGMA}\n")
                f_cross.flush()

        f_in.close();
        f_bar.close();
        f_cross.close();
        f_parti.close();
        f_trans.close()

    def pot_shape(self):
        # Calculate Potentials (same logic as before, just cleaner class access)
        if self.L == 0:
            if self.Betap >= 0 and self.Betat >= 0:
                T1, T2 = 0., 0.
            elif self.Betap < 0 and self.Betat >= 0:
                T1, T2 = self.p[17] / 2., 0.
            elif self.Betap >= 0 and self.Betat < 0:
                T1, T2 = 0., self.p[17] / 2.
            else:
                T1, T2 = self.p[17] / 2., self.p[17] / 2.

            # --------------------------------
            # NEW: Brentq for Tip-Tip Barrier
            # --------------------------------
            def deriv_wrapper_tip(d):
                return DV_total_jit(d, self.Betap, self.Betat, T1, T2, 0, self.p, self.gl_nodes, self.gl_weights)

            dis = 5.0
            # 1. Bracket Finding
            while True:
                dis -= 1.0
                if dis < -10.0: break
                if deriv_wrapper_tip(dis) * deriv_wrapper_tip(dis - 1.0) < 0:
                    break

            # 2. Root Finding
            try:
                R1 = scipy.optimize.brentq(deriv_wrapper_tip, dis - 1.0, dis)
            except ValueError:
                R1 = dis - 0.5

            self.VBtip = V_total_jit(R1, self.Betap, self.Betat, T1, T2, 0, self.p, self.gl_nodes, self.gl_weights)

        if self.Betap >= 0 and self.Betat >= 0:
            T1, T2 = self.p[17] / 2., self.p[17] / 2.
        else:
            T1, T2 = 0., 0.

        # --------------------------------
        # NEW: Brentq for Waist-Waist Barrier
        # --------------------------------
        def deriv_wrapper_waist(d):
            return DV_total_jit(d, self.Betap, self.Betat, T1, T2, 0, self.p, self.gl_nodes, self.gl_weights)

        dis = 5.0
        # 1. Bracket Finding
        while True:
            dis -= 1.0
            if dis < -5.0: break
            if deriv_wrapper_waist(dis) * deriv_wrapper_waist(dis - 1.0) < 0:
                break

        # 2. Root Finding
        try:
            R1 = scipy.optimize.brentq(deriv_wrapper_waist, dis - 1.0, dis)
        except ValueError:
            R1 = dis - 0.5

        self.RB = calc_R_scalar(R1, self.Betap, self.Betat, T1, T2, self.p[4], self.p[5], self.p[12])
        self.VB = V_total_jit(R1, self.Betap, self.Betat, T1, T2, 0, self.p, self.gl_nodes, self.gl_weights)

        ddv = (DV_total_jit(R1 + 1e-5, self.Betap, self.Betat, T1, T2, 0, self.p, self.gl_nodes, self.gl_weights) -
               DV_total_jit(R1 - 1e-5, self.Betap, self.Betat, T1, T2, 0, self.p, self.gl_nodes,
                            self.gl_weights)) / 2e-5
        self.CURV = self.p[16] * math.sqrt(abs(ddv) / self.p[6])

        if self.L == 0:
            self.RB0 = self.RB
            self.VB0 = self.VB
            Betai, Betaf, Dbet = -1.0, 2.0, 0.05
            K = int((Betaf - Betai) / Dbet) + 1
            args_list =[]
            for I1 in range(K):
                # 【修改】：加入之前计算得出的物理取向 T1, T2
                args_list.append((I1, Betai, Dbet, self.Betap, self.Betat,
                                  self.p[0], self.p[8], self.p[2], self.p[7], self.p,
                                  self.gl_nodes, self.gl_weights, T1, T2))

            print(f"  > Calculating potential surface ({K} points) in parallel...")
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(compute_potential_surface_point, args_list))
            results.sort(key=lambda x: x[0])
            VSA, BET, BETP, BETT = [], [], [],[]
            for res in results:
                _, vs_val, beta, b1, b2 = res
                VSA.append(vs_val)
                BET.append(beta)
                BETP.append(b1)
                BETT.append(b2)
            self.VS = 1000.0
            idx = 0
            for i, vt in enumerate(VSA):
                if self.VS > vt and vt != 0.0:
                    self.VS = vt
                    idx = i
            self.BETAS = BET[idx]
            self.BTP = BETP[idx]
            self.BTT = BETT[idx]

    def precompute_bdf(self):
        """
        Computes the Barrier Distribution Function (BDF) array once.
        This replaces the O(N^2) loops in the original code.
        """
        self.EDF = (self.p[7] * self.BTP ** 2 + self.p[8] * self.BTT ** 2) / 2.0
        VM = self.barr1 * self.VB0 + (1. - self.barr1) * self.VS
        DELT1 = self.barr2 * self.EDF
        DELT2 = (1.24 - self.barr2) * self.EDF

        VMIN = VM - 5.0 * DELT1
        VMAX = VM + 5.0 * DELT2
        DELV = 0.05
        N = int((VMAX - VMIN) / DELV) + 1

        # Vectorized Grid
        self.V_grid = np.linspace(VMIN, VMAX, N)

        # Vectorized BDF Calculation
        mask_le = self.V_grid <= VM
        mask_gt = ~mask_le

        ce_vals = np.empty_like(self.V_grid)
        ce_vals[mask_le] = ((self.V_grid[mask_le] - VM) / DELT1) ** 2
        ce_vals[mask_gt] = ((self.V_grid[mask_gt] - VM) / DELT2) ** 2

        # Calculate unnormalized probabilities
        exp_ce = np.exp(-ce_vals)
        FB = np.sum(exp_ce) * DELV

        if FB == 0.0: FB = 1.0

        self.BDF_weighted = exp_ce * DELV / FB


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    start_time = time.time()
    model = EMCCModel()
    model.run()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    input("Press Enter to exit...")