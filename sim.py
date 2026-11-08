import numpy as np
import numba as nb
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cv2


def annulus_mesh(r_in, r_out, n_theta=48, n_r=4):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rs = np.linspace(r_in, r_out, n_r + 1)
    X = np.array([[r * np.cos(t), r * np.sin(t)] for r in rs for t in th])
    tris = []
    for j in range(n_r):
        for i in range(n_theta):
            a = j * n_theta + i
            b = j * n_theta + (i + 1) % n_theta
            c = (j + 1) * n_theta + i
            d = (j + 1) * n_theta + (i + 1) % n_theta
            if (i + j) % 2 == 0:
                tris += [[a, b, d], [a, d, c]]
            else:
                tris += [[a, b, c], [b, d, c]]
    tris = np.array(tris, dtype=np.int64)
    inner = np.arange(n_theta)
    outer = np.arange(n_r * n_theta, (n_r + 1) * n_theta)
    return X, tris, inner, outer


@nb.njit(cache=True)
def _rot(F):
    a = np.arctan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])
    c, s = np.cos(a), np.sin(a)
    R = np.empty((2, 2))
    R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R


@nb.njit(cache=True)
def _element_pass(x, v, tris, Dminv, Bm, X, Kunit, Cunit, A0h, eps_v,
                  E, Em, eta, coef_K, coef_C, f_out, Adata, eps_out):
    ne = tris.shape[0]
    for e in range(ne):
        i0, i1, i2 = tris[e, 0], tris[e, 1], tris[e, 2]
        Ds = np.empty((2, 2))
        Ds[0, 0] = x[i1, 0] - x[i0, 0]; Ds[1, 0] = x[i1, 1] - x[i0, 1]
        Ds[0, 1] = x[i2, 0] - x[i0, 0]; Ds[1, 1] = x[i2, 1] - x[i0, 1]
        F = Ds @ Dminv[e]
        R = _rot(F)
        u = np.empty(6)
        w = np.empty(6)
        for k in range(3):
            n = tris[e, k]
            px = R[0, 0] * x[n, 0] + R[1, 0] * x[n, 1]
            py = R[0, 1] * x[n, 0] + R[1, 1] * x[n, 1]
            u[2 * k] = px - X[n, 0]
            u[2 * k + 1] = py - X[n, 1]
            w[2 * k] = R[0, 0] * v[n, 0] + R[1, 0] * v[n, 1]
            w[2 * k + 1] = R[0, 1] * v[n, 0] + R[1, 1] * v[n, 1]
        eps = Bm[e] @ u
        eps_out[e, :] = eps
        s_el = E * (Cunit @ eps) + Em * (Cunit @ (eps - eps_v[e]))
        f_loc = -A0h[e] * (Bm[e].T @ s_el)
        for k in range(3):
            n = tris[e, k]
            fx = R[0, 0] * f_loc[2 * k] + R[0, 1] * f_loc[2 * k + 1]
            fy = R[1, 0] * f_loc[2 * k] + R[1, 1] * f_loc[2 * k + 1]
            f_out[n, 0] += fx
            f_out[n, 1] += fy
        K = Kunit[e]
        cK = coef_K * (E + Em) + coef_C * eta
        for a in range(3):
            for b in range(3):
                k00 = K[2 * a, 2 * b]; k01 = K[2 * a, 2 * b + 1]
                k10 = K[2 * a + 1, 2 * b]; k11 = K[2 * a + 1, 2 * b + 1]
                t00 = R[0, 0] * k00 + R[0, 1] * k10
                t01 = R[0, 0] * k01 + R[0, 1] * k11
                t10 = R[1, 0] * k00 + R[1, 1] * k10
                t11 = R[1, 0] * k01 + R[1, 1] * k11
                r00 = t00 * R[0, 0] + t01 * R[0, 1]
                r01 = t00 * R[1, 0] + t01 * R[1, 1]
                r10 = t10 * R[0, 0] + t11 * R[0, 1]
                r11 = t10 * R[1, 0] + t11 * R[1, 1]
                base = e * 36 + (2 * a) * 6 + 2 * b
                Adata[base] = cK * r00
                Adata[base + 1] = cK * r01
                Adata[base + 6] = cK * r10
                Adata[base + 7] = cK * r11


@nb.njit(cache=True)
def _strain_only(x, tris, Dminv, Bm, X, eps_out):
    ne = tris.shape[0]
    for e in range(ne):
        i0, i1, i2 = tris[e, 0], tris[e, 1], tris[e, 2]
        Ds = np.empty((2, 2))
        Ds[0, 0] = x[i1, 0] - x[i0, 0]; Ds[1, 0] = x[i1, 1] - x[i0, 1]
        Ds[0, 1] = x[i2, 0] - x[i0, 0]; Ds[1, 1] = x[i2, 1] - x[i0, 1]
        F = Ds @ Dminv[e]
        R = _rot(F)
        u = np.empty(6)
        for k in range(3):
            n = tris[e, k]
            u[2 * k] = R[0, 0] * x[n, 0] + R[1, 0] * x[n, 1] - X[n, 0]
            u[2 * k + 1] = R[0, 1] * x[n, 0] + R[1, 1] * x[n, 1] - X[n, 1]
        eps_out[e, :] = Bm[e] @ u


class RingSim:
    def __init__(self, r_in=0.0225, r_out=0.043, height=0.021, mass=0.075,
                 n_theta=48, n_r=4, finger_r=0.008, table_drag=2.0):
        self.X, self.tris, self.inner, self.outer = annulus_mesh(r_in, r_out, n_theta, n_r)
        self.n = len(self.X)
        self.h = height
        self.finger_r = finger_r
        X, T = self.X, self.tris
        ne = len(T)
        Dm = np.stack([np.stack([X[T[:, 1], 0] - X[T[:, 0], 0], X[T[:, 2], 0] - X[T[:, 0], 0]], 1),
                       np.stack([X[T[:, 1], 1] - X[T[:, 0], 1], X[T[:, 2], 1] - X[T[:, 0], 1]], 1)], 1)
        self.Dminv = np.linalg.inv(Dm)
        A0 = 0.5 * np.abs(np.linalg.det(Dm))
        self.A0h = A0 * height
        Bm = np.zeros((ne, 3, 6))
        for e in range(ne):
            g1, g2 = self.Dminv[e][0], self.Dminv[e][1]
            g0 = -(g1 + g2)
            for k, g in enumerate((g0, g1, g2)):
                Bm[e, 0, 2 * k] = g[0]
                Bm[e, 1, 2 * k + 1] = g[1]
                Bm[e, 2, 2 * k] = g[1]
                Bm[e, 2, 2 * k + 1] = g[0]
        self.Bm = Bm
        node_area = np.zeros(self.n)
        for e in range(ne):
            node_area[T[e]] += A0[e] / 3
        self.m = mass * node_area / node_area.sum()
        self.table_drag = table_drag
        dof = np.stack([2 * T[:, k] + d for k in range(3) for d in range(2)], 1)
        rows = np.repeat(dof, 6, axis=1).ravel()
        cols = np.tile(dof, (1, 6)).ravel()
        N = 2 * self.n
        ones = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsc()
        ones.sum_duplicates()
        self._indptr, self._indices = ones.indptr, ones.indices
        key = cols.astype(np.int64) * N + rows
        csc_keys = np.repeat(np.arange(N), np.diff(ones.indptr)).astype(np.int64) * N + ones.indices
        order = np.argsort(csc_keys)
        self._slot = order[np.searchsorted(csc_keys[order], key)]
        self._nnz = len(ones.indices)
        self._diag_slot = np.array([self._slot[np.where((rows == i) & (cols == i))[0][0]] for i in range(N)])
        self.set_params()
        self.reset()

    def set_params(self, E=3e6, nu=0.45, eta=3e4, beta=0.3, tau_m=0.2):
        self.E, self.nu, self.eta, self.beta, self.tau_m = E, nu, eta, beta, tau_m
        Cu = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu ** 2)
        self.Cunit = Cu
        self.Kunit = np.einsum("eij,jk,ekl->eil", self.Bm.transpose(0, 2, 1), Cu, self.Bm) \
            * self.A0h[:, None, None]

    def reset(self):
        self.x = self.X.copy()
        self.v = np.zeros_like(self.X)
        self.eps_v = np.zeros((len(self.tris), 3))
        self.gap = None
        self.active = np.zeros(self.n, bool)
        self.owner = np.zeros(self.n, np.int64)
        self.contact_force = np.zeros((self.n, 2))
        self.t = 0.0

    def _finger_centres(self, gap):
        r = self.finger_r
        return np.array([[0.0, gap / 2 + r], [0.0, -gap / 2 - r]])

    def _assemble(self, dt):
        n = self.n
        f = np.zeros((n, 2))
        Adata = np.zeros(len(self.tris) * 36)
        eps = np.zeros((len(self.tris), 3))
        _element_pass(self.x, self.v, self.tris, self.Dminv, self.Bm, self.X,
                      self.Kunit, self.Cunit, self.A0h, self.eps_v,
                      self.E, self.beta * self.E, self.eta, dt * dt, dt, f, Adata, eps)
        data = np.bincount(self._slot, weights=Adata, minlength=self._nnz)
        mdiag = np.repeat(self.m, 2)
        data[self._diag_slot] += mdiag * (1.0 + dt * self.table_drag)
        A = sp.csc_matrix((data, self._indices, self._indptr), shape=(2 * n, 2 * n))
        rhs = mdiag * self.v.ravel() + dt * f.ravel()
        return A, rhs

    @staticmethod
    def _solve_constrained(A, rhs, fixed_dofs, v_fixed):
        N = A.shape[0]
        if len(fixed_dofs) == 0:
            return spla.spsolve(A, rhs)
        free = np.ones(N, bool)
        free[fixed_dofs] = False
        fi = np.where(free)[0]
        v = np.zeros(N)
        v[fixed_dofs] = v_fixed
        Aff = A[fi][:, fi]
        b = rhs[fi] - A[fi][:, fixed_dofs] @ v_fixed
        v[fi] = spla.spsolve(Aff.tocsc(), b)
        return v

    def step(self, dt, gap=None):
        n = self.n
        A, rhs = self._assemble(dt)
        if gap is None:
            self.active[:] = False
            v = spla.spsolve(A, rhs)
            self.contact_force = np.zeros((n, 2))
        else:
            C = self._finger_centres(gap)
            C_prev = self._finger_centres(self.gap) if self.gap is not None else C
            dC = C - C_prev
            active, owner = self.active.copy(), self.owner.copy()
            target = np.zeros((n, 2))
            for k in range(2):
                on = active & (owner == k)
                target[on] = self.x[on] + dC[k]
            for _ in range(8):
                dofs = np.where(np.repeat(active, 2))[0]
                vfix = ((target[active] - self.x[active]) / dt).ravel()
                v = self._solve_constrained(A, rhs, dofs, vfix)
                xn = self.x + dt * v.reshape(n, 2)
                fc = ((A @ v - rhs) / dt).reshape(n, 2)
                changed = False
                for k in range(2):
                    d = xn - C[k]
                    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
                    nrm = d / dist[:, None]
                    new = (~active) & (dist < self.finger_r)
                    if new.any():
                        active[new], owner[new] = True, k
                        target[new] = C[k] + self.finger_r * nrm[new]
                        changed = True
                    pull = active & (owner == k) & (np.sum(fc * nrm, axis=1) < 0.0)
                    if pull.any():
                        active[pull] = False
                        changed = True
                if not changed:
                    break
            self.active, self.owner = active, owner
            self.contact_force = ((A @ v - rhs) / dt).reshape(n, 2)
        self.v = v.reshape(n, 2)
        self.x = self.x + dt * self.v
        self.gap = gap
        eps = np.zeros((len(self.tris), 3))
        _strain_only(self.x, self.tris, self.Dminv, self.Bm, self.X, eps)
        a = np.exp(-dt / self.tau_m)
        self.eps_v = eps + (self.eps_v - eps) * a
        self.t += dt

    def hole_axes(self):
        pts = (self.x[self.inner] * 1e5).astype(np.float32)
        (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts)
        return min(d1, d2) / 1e5, max(d1, d2) / 1e5

    def finger_force(self):
        on = self.active & (self.owner == 0)
        return float(-self.contact_force[on, 1].sum())

    def settle(self, gap, iters=30, dt=0.05):
        for _ in range(iters):
            self.step(dt, gap)
            self.v[:] = 0.0
            self.eps_v[:] = 0.0
        return self.hole_axes()
