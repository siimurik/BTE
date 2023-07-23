import numpy as np
cimport numpy as cnp

cpdef bicgstab_(cnp.ndarray x0, cnp.ndarray b, atv, cnp.ndarray params):
    cdef:
        int n = b.shape[0]
        double errtol = params[0] * np.linalg.norm(b)
        int kmax = params[1]
        list error = []
        cnp.ndarray x = x0.copy()
        cnp.ndarray r = np.empty(n)
        cnp.ndarray hatr0 = np.empty(n)
        double rho_k, rho_k_minus_1, alpha, omega, beta, tau, zeta
        cnp.ndarray v = np.empty(n)
        cnp.ndarray p = np.empty(n)
        int k = 0
        double tau_squared, omega_tau_squared

    if np.linalg.norm(x) != 0:
        r = b - atv(x)
    else:
        r = b

    hatr0[:] = r
    rho_k = 1
    alpha = 1
    omega = 1
    v[:] = 0
    p[:] = 0
    rho_k_minus_1 = 1
    zeta = np.linalg.norm(r)
    error.append(zeta)

    # Bi-CGSTAB iteration
    total_iters = 0
    while zeta > errtol and k < kmax:
        k += 1
        if omega == 0:
            raise ValueError("Bi-CGSTAB breakdown, omega=0")

        beta = (rho_k / rho_k_minus_1) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = atv(p)
        tau = np.dot(hatr0, v)

        if tau == 0:
            raise ValueError("Bi-CGSTAB breakdown, tau=0")

        alpha = rho_k / tau
        r -= alpha * v
        s = r.copy()
        t = atv(s)
        tau_squared = np.dot(t, t)

        if tau_squared == 0:
            raise ValueError("Bi-CGSTAB breakdown, tau_squared=0")

        omega_tau_squared = np.dot(t, s) / tau_squared
        omega = omega_tau_squared * tau / tau_squared
        rho_k = -omega * np.dot(hatr0, t)
        x += alpha * p + omega * s
        zeta = np.linalg.norm(r)
        total_iters = k
        error.append(zeta)

    return x, np.array(error), total_iters
