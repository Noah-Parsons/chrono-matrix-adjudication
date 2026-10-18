"""
Chrono (PyChrono 10.0.0) as the fourth engine -- matrix-method adapter.

WHY THIS ROUTE
--------------
The obvious call, system.DoAssembly(0xFFFF) followed by body.GetPosDt2(),
does not return the constrained acceleration: at the acceleration level the
assembly analysis reports a velocity-increment measure inherited from Chrono's
contact-dynamics formulation, which omits the quadratic-velocity constraint
term (projectchrono/chrono#846). The routine that does solve the
acceleration-level problem, StateSolveA, cannot be reached from Python because
ChState / ChStateDelta lack an Eigen::Index typemap (projectchrono/chrono#847).

The constrained solution is nevertheless recoverable from Chrono's OWN
matrices, as D. Negrut described (30 Sep 2026):

    | M   Cq^T | | a   |   | f  |
    | Cq  0    | | lam | = | Qc |,     Qc = -(dCq/dt) v

Chrono supplies M and Cq at the current state (ChSystem::WriteSystemMatrices;
WriteMatrixBlocks is unreachable on this build because GetSystemDescriptor()
returns an unwrapped shared_ptr). The adapter supplies f (gravity) and Qc,
then solves the saddle-point system in numpy. Constraint forces are compared
as Cq^T lam, never as lam, since multipliers are scaled by convention.

MODELLING
---------
Every family in the study is built from point masses, massless rods and
prismatic slides, so each is modelled in Chrono with exactly those pieces:

  * a point mass is a ChBody with a small isotropic inertia; every attachment
    is at the body's centre, so rotations are decoupled and stay at zero;
  * a massless rod is a ChLinkDistance between two body centres (or a body
    centre and the ground origin);
  * a slide along x is a ChLinkLockPrismatic whose free axis (link z) is
    rotated onto world x.

Qc for a distance row is analytic: if Chrono's row is k * [-r^, +r^] on the
two bodies' linear blocks, then (dCq/dt) v = k |v_rel,perp|^2 / d. For the
prismatic rows Qc = 0, because the ground is fixed, the attachment is at the
body centre, and the body does not rotate. Both claims are CHECKED at every
probe: Cq_check measures how far each distance row departs from the assumed
form, and Qc_fd compares the analytic Qc with a central difference of
Chrono's own Cq along the velocity.

The state is applied with Setup() + Update(), not DoAssembly(), so Chrono does
not move it; the state is read back regardless and the reference is evaluated
at the read-back state.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np

ROT_INERTIA = 1e-2      # per unit mass; irrelevant to the result (decoupled)
NATIVE_SOLVER = os.environ.get("CHRONO_NATIVE_SOLVER", "SPARSE_QR")
"""Solver for the DoAssembly contrast column only; the matrix method uses none.
Set CHRONO_NATIVE_SOLVER=PSOR to reproduce the original default-solver run."""


def relerr(a, r):
    """The study's metric: max |a - r| / max(|r|, 1), elementwise."""
    a = np.asarray(a, float)
    r = np.asarray(r, float)
    return float(np.max(np.abs(a - r) / np.maximum(np.abs(r), 1.0)))


def _load_coo(path):
    """Chrono's COO dump, 1-based (WriteSystemMatrices(..., one_indexed=True))."""
    data = np.loadtxt(path, ndmin=2)
    nr = int(data[:, 0].max())
    nc = int(data[:, 1].max())
    A = np.zeros((nr, nc))
    for r, c, val in data:
        A[int(r) - 1, int(c) - 1] += val
    return A


# ===========================================================================
# The Chrono model
# ===========================================================================

class ChronoModel:
    """A Chrono system of point-mass bodies, with the bookkeeping the matrix
    method needs: body order, masses, distance links, prismatic slides."""

    def __init__(self, g: float) -> None:
        import pychrono as chrono
        self.ch = chrono
        self.sys = chrono.ChSystemNSC()
        self.sys.SetSolverType(getattr(chrono.ChSolver, f"Type_{NATIVE_SOLVER}"))
        self.sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -g, 0))
        self.g = np.array([0.0, -g, 0.0])
        self.ground = chrono.ChBody()
        self.ground.SetFixed(True)
        self.sys.Add(self.ground)
        self.bodies, self.masses = [], []
        self.distances = []          # (i, j, L); i == -1 is the ground origin
        self.prismatic = []          # body indices confined to the x-axis
        self._keep = []              # keep Python references to the links
        self.refreshed_by = None

    def _v(self, a):
        return self.ch.ChVector3d(float(a[0]), float(a[1]), float(a[2]))

    @staticmethod
    def _np(v):
        return np.array([v.x, v.y, v.z], float)

    def body(self, m, p, v):
        b = self.ch.ChBody()
        b.SetMass(float(m))
        inertia = ROT_INERTIA * float(m)
        b.SetInertiaXX(self.ch.ChVector3d(inertia, inertia, inertia))
        b.SetPos(self._v(p))
        b.SetPosDt(self._v(v))
        self.sys.Add(b)
        self.bodies.append(b)
        self.masses.append(float(m))
        return len(self.bodies) - 1

    def distance(self, i, j, L):
        bi = self.ground if i < 0 else self.bodies[i]
        pi = np.zeros(3) if i < 0 else self._np(self.bodies[i].GetPos())
        pj = self._np(self.bodies[j].GetPos())
        link = self.ch.ChLinkDistance()
        link.Initialize(bi, self.bodies[j], False, self._v(pi), self._v(pj), True)
        self.sys.AddLink(link)
        self._keep.append(link)
        self.distances.append((i, j, float(L)))

    def slide_x(self, j):
        ch = self.ch
        if hasattr(ch, "QuatFromAngleY"):
            q = ch.QuatFromAngleY(math.pi / 2)      # link z -> world x
        else:
            q = ch.Q_ROTATE_Z_TO_X
        p = self._np(self.bodies[j].GetPos())
        link = ch.ChLinkLockPrismatic()
        link.Initialize(self.ground, self.bodies[j], ch.ChFramed(self._v(p), q))
        self.sys.AddLink(link)
        self._keep.append(link)
        self.prismatic.append(j)

    def refresh(self, assembly: bool = False) -> str:
        """Bring Chrono's internal quantities (Cq in particular) up to date."""
        if assembly:
            self.sys.DoAssembly(0xFFFF)
            self.refreshed_by = "DoAssembly"
            return self.refreshed_by
        try:
            self.sys.Setup()
        except Exception:
            pass
        for args in ((True,), ()):
            try:
                self.sys.Update(*args)
                self.refreshed_by = "Update"
                return self.refreshed_by
            except Exception:
                continue
        self.sys.DoAssembly(0xFFFF)
        self.refreshed_by = "DoAssembly (Update unavailable)"
        return self.refreshed_by

    def state(self):
        P = np.array([self._np(b.GetPos()) for b in self.bodies])
        V = np.array([self._np(b.GetPosDt()) for b in self.bodies])
        return P, V

    def native_acc(self):
        return np.array([self._np(b.GetPosDt2()) for b in self.bodies])

    def matrices(self):
        d = tempfile.mkdtemp(prefix="chrono_mm_")
        try:
            self.sys.WriteSystemMatrices(True, False, False, True,
                                         os.path.join(d, "solve"), True)
            M = _load_coo(os.path.join(d, "solve_M.dat"))
            Cq = _load_coo(os.path.join(d, "solve_Cq.dat"))
        finally:
            shutil.rmtree(d, ignore_errors=True)
        n = 6 * len(self.bodies)
        if M.shape != (n, n):
            raise RuntimeError(f"M is {M.shape}, expected {(n, n)} "
                               f"({len(self.bodies)} movable bodies)")
        if Cq.shape[1] < n:
            Cq = np.hstack([Cq, np.zeros((Cq.shape[0], n - Cq.shape[1]))])
        return M, Cq


# ===========================================================================
# The matrix method
# ===========================================================================

def _full(nb, X):
    """Body-level 3-vectors -> generalised 6nb vector (rotational parts zero)."""
    out = np.zeros(6 * nb)
    for b in range(nb):
        out[6 * b:6 * b + 3] = X[b]
    return out


def qc_analytic(model, Cq, P, V):
    """Qc = -(dCq/dt) v. Distance rows analytic; all other rows zero."""
    nb = len(model.bodies)
    n, m = 6 * nb, Cq.shape[0]
    Qc = np.zeros(m)
    matched, misalign = {}, 0.0
    for k in range(m):
        row = Cq[k]
        nrow = np.linalg.norm(row)
        if nrow == 0.0:
            continue
        for idx, (i, j, _L) in enumerate(model.distances):
            if idx in matched:
                continue
            pi = np.zeros(3) if i < 0 else P[i]
            vi = np.zeros(3) if i < 0 else V[i]
            r = P[j] - pi
            d = np.linalg.norm(r)
            rh = r / d
            u = np.zeros(n)
            u[6 * j:6 * j + 3] = rh
            if i >= 0:
                u[6 * i:6 * i + 3] = -rh
            if abs(row @ u) / (nrow * np.linalg.norm(u)) > 1.0 - 1e-9:
                kk = (row @ u) / (u @ u)
                misalign = max(misalign, np.linalg.norm(row - kk * u) / abs(kk))
                vrel = V[j] - vi
                vperp2 = vrel @ vrel - (rh @ vrel) ** 2
                Qc[k] = -kk * vperp2 / d
                matched[idx] = k
                break
    return Qc, matched, misalign


def solve(model, P, V):
    M, Cq = model.matrices()
    nb = len(model.bodies)
    n, m = 6 * nb, Cq.shape[0]

    expect = np.concatenate([[mi] * 3 + [ROT_INERTIA * mi] * 3
                             for mi in model.masses])
    M_check = max(float(np.max(np.abs(np.diag(M) - expect) / expect)),
                  float(np.max(np.abs(M - np.diag(np.diag(M))))))

    Qc, matched, misalign = qc_analytic(model, Cq, P, V)
    if len(matched) != len(model.distances):
        raise RuntimeError(f"matched {len(matched)} of {len(model.distances)} "
                           f"distance links to rows of Cq ({m} rows)")

    f = _full(nb, np.array(model.masses)[:, None] * model.g)
    K = np.block([[M, Cq.T], [Cq, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.concatenate([f, Qc]))
    a, lam = sol[:n], sol[n:]
    Fc = -(Cq.T @ lam)                                   # compare Cq^T lam
    return {
        "A": a.reshape(nb, 6)[:, :3], "alpha": a.reshape(nb, 6)[:, 3:],
        "F": Fc.reshape(nb, 6)[:, :3],
        "M": M, "Cq": Cq, "Qc": Qc, "f": f, "n": n, "m": m,
        "M_check": M_check, "Cq_check": misalign,
        "rows": m, "distance_rows": len(matched),
    }


def exact_residual(sol, A_ex):
    """Closed-form accelerations substituted into Chrono's assembled equations."""
    nb = sol["n"] // 6
    a = _full(nb, A_ex)
    M, Cq, f, Qc = sol["M"], sol["Cq"], sol["f"], sol["Qc"]
    lam = np.linalg.lstsq(Cq.T, f - M @ a, rcond=None)[0]
    res_dyn = (np.linalg.norm(M @ a + Cq.T @ lam - f)
               / max(np.linalg.norm(f), np.linalg.norm(M @ a), 1.0))
    res_con = np.linalg.norm(Cq @ a - Qc) / max(np.linalg.norm(Qc), 1.0)
    return float(res_dyn), float(res_con)


def guard(model, P, V, A_ex):
    """Reference guard: C, dC/dt and d2C/dt2 of the reference at this state."""
    worst = 0.0
    for i, j, L in model.distances:
        pi = np.zeros(3) if i < 0 else P[i]
        vi = np.zeros(3) if i < 0 else V[i]
        ai = np.zeros(3) if i < 0 else A_ex[i]
        r, vr, ar = P[j] - pi, V[j] - vi, A_ex[j] - ai
        d = np.linalg.norm(r)
        rh = r / d
        vperp2 = vr @ vr - (rh @ vr) ** 2
        worst = max(worst, abs(d - L), abs(rh @ vr), abs(rh @ ar + vperp2 / d))
    for j in model.prismatic:
        worst = max(worst, abs(P[j][1]), abs(P[j][2]), abs(V[j][1]),
                    abs(V[j][2]), abs(A_ex[j][1]), abs(A_ex[j][2]))
    return float(worst)


# ===========================================================================
# The three families. Each maps the study's generalised state to point-mass
# positions and velocities, builds the Chrono model, and maps accelerations
# back. Conventions follow the references exactly.
# ===========================================================================

def _t(th):          # unit tangent for an angle from the downward vertical
    return np.array([math.cos(th), math.sin(th), 0.0])


def _e(th):          # unit vector from pivot to mass, angle from downward vertical
    return np.array([math.sin(th), -math.cos(th), 0.0])


class Family:
    model_cls = ChronoModel

    def build(self, P, V):
        raise NotImplementedError


class ChainFamily(Family):
    """reference.NLinkChain: absolute angles from the downward vertical,
    interleaved state [th0, w0, th1, w1, ...]."""

    def __init__(self, ref):
        self.ref, self.N, self.l = ref, ref.N, ref.l

    def kin(self, s):
        th, w = s[0::2], s[1::2]
        P = np.cumsum([self.l * _e(t) for t in th], axis=0)
        V = np.cumsum([self.l * wi * _t(t) for t, wi in zip(th, w)], axis=0)
        return P, V

    def build(self, P, V):
        mdl = self.model_cls(self.ref.g)
        for b in range(self.N):
            mdl.body(self.ref.m, P[b], V[b])
        mdl.distance(-1, 0, self.l)
        for b in range(1, self.N):
            mdl.distance(b - 1, b, self.l)
        return mdl

    def _rel(self, X):
        return np.diff(np.vstack([np.zeros(3), X]), axis=0)

    def generalised(self, P, V):
        E, Ed = self._rel(P), self._rel(V)
        s = np.empty(2 * self.N)
        for b in range(self.N):
            th = math.atan2(E[b][0], -E[b][1])
            s[2 * b] = th
            s[2 * b + 1] = _t(th) @ Ed[b] / np.linalg.norm(E[b])
        return s

    def to_qdd(self, P, V, A):
        E, Ea = self._rel(P), self._rel(A)
        return np.array([_t(math.atan2(E[b][0], -E[b][1])) @ Ea[b]
                         / np.linalg.norm(E[b]) for b in range(self.N)])

    def ref_accel(self, s):
        return self.ref.accel(s)

    def body_acc(self, s, qdd):
        th, w = s[0::2], s[1::2]
        rel = [self.l * (qdd[b] * _t(th[b]) - w[b] ** 2 * _e(th[b]))
               for b in range(self.N)]
        return np.cumsum(rel, axis=0)


class SliderCrankFamily(Family):
    """reference_slidercrank.SliderCrank: q = (theta, x), state
    [th, thd, x, xd]; crank a point mass at the pin, slider on the x-axis."""

    def __init__(self, ref):
        self.ref = ref

    def kin(self, s):
        th, thd, x, xd = s
        r = self.ref.r
        P = np.array([[r * math.cos(th), r * math.sin(th), 0.0],
                      [x, 0.0, 0.0]])
        V = np.array([[-r * thd * math.sin(th), r * thd * math.cos(th), 0.0],
                      [xd, 0.0, 0.0]])
        return P, V

    def build(self, P, V):
        mdl = self.model_cls(self.ref.g)
        mdl.body(self.ref.m_c, P[0], V[0])
        mdl.body(self.ref.m_s, P[1], V[1])
        mdl.distance(-1, 0, self.ref.r)        # crank: pin at radius r
        mdl.distance(0, 1, self.ref.l)         # massless connecting rod
        mdl.slide_x(1)                         # slider on the x-axis
        return mdl

    def generalised(self, P, V):
        th = math.atan2(P[0][1], P[0][0])
        perp = np.array([-math.sin(th), math.cos(th), 0.0])
        return np.array([th, perp @ V[0] / np.linalg.norm(P[0][:2]),
                         P[1][0], V[1][0]])

    def to_qdd(self, P, V, A):
        th = math.atan2(P[0][1], P[0][0])
        perp = np.array([-math.sin(th), math.cos(th), 0.0])
        return np.array([perp @ A[0] / np.linalg.norm(P[0][:2]), A[1][0]])

    def ref_accel(self, s):
        return self.ref.accel(s)

    def body_acc(self, s, qdd):
        th, thd = s[0], s[1]
        r = self.ref.r
        c, sn = math.cos(th), math.sin(th)
        return np.array([[r * (-qdd[0] * sn - thd ** 2 * c),
                          r * (qdd[0] * c - thd ** 2 * sn), 0.0],
                         [qdd[1], 0.0, 0.0]])


class CartPoleFamily(Family):
    """reference_cartpole.CartPole: q = (x, theta), state [x, xd, th, thd];
    point mass at x + l sin th, -l cos th."""

    def __init__(self, ref):
        self.ref = ref

    def kin(self, s):
        x, xd, th, thd = s
        l = self.ref.l
        C = np.array([x, 0.0, 0.0])
        Cv = np.array([xd, 0.0, 0.0])
        return (np.array([C, C + l * _e(th)]),
                np.array([Cv, Cv + l * thd * _t(th)]))

    def build(self, P, V):
        mdl = self.model_cls(self.ref.g)
        mdl.body(self.ref.M, P[0], V[0])       # cart
        mdl.body(self.ref.m, P[1], V[1])       # pole mass
        mdl.slide_x(0)
        mdl.distance(0, 1, self.ref.l)
        return mdl

    def generalised(self, P, V):
        E, Ed = P[1] - P[0], V[1] - V[0]
        th = math.atan2(E[0], -E[1])
        return np.array([P[0][0], V[0][0], th,
                         _t(th) @ Ed / np.linalg.norm(E)])

    def to_qdd(self, P, V, A):
        E = P[1] - P[0]
        th = math.atan2(E[0], -E[1])
        return np.array([A[0][0], _t(th) @ (A[1] - A[0]) / np.linalg.norm(E)])

    def ref_accel(self, s):
        return self.ref.accel(s)

    def body_acc(self, s, qdd):
        th, thd = s[2], s[3]
        l = self.ref.l
        A0 = np.array([qdd[0], 0.0, 0.0])
        return np.array([A0, A0 + l * (qdd[1] * _t(th) - thd ** 2 * _e(th))])


# ===========================================================================
# One probe
# ===========================================================================

def probe(fam, s, assembly=False, fd_h=1e-5, native=True):
    s = np.asarray(s, float)
    P0, V0 = fam.kin(s)
    mdl = fam.build(P0, V0)
    how = mdl.refresh(assembly)
    P, V = mdl.state()
    shift = float(max(np.abs(P - P0).max(), np.abs(V - V0).max()))

    st = fam.generalised(P, V)
    q_ref = np.asarray(fam.ref_accel(st), float)
    A_ex = fam.body_acc(st, q_ref)
    masses = np.array(mdl.masses)
    F_ex = masses[:, None] * (A_ex - mdl.g)

    sol = solve(mdl, P, V)
    q_mm = fam.to_qdd(P, V, sol["A"])
    out = {
        "refresh": how, "shift": shift,
        "gen": relerr(q_mm, q_ref),                  # the study's metric
        "body": relerr(sol["A"], A_ex),
        "force": relerr(sol["F"], F_ex),
        "alpha": float(np.abs(sol["alpha"]).max()),
        "guard": guard(mdl, P, V, A_ex),
        "M_check": sol["M_check"], "Cq_check": sol["Cq_check"],
        "rows": sol["rows"], "distance_rows": sol["distance_rows"],
    }
    out["res_dyn"], out["res_con"] = exact_residual(sol, A_ex)

    if fd_h:
        nb = len(mdl.bodies)
        Cqs = []
        for sign in (+1.0, -1.0):
            m2 = fam.build(P + sign * fd_h * V, V)
            m2.refresh(assembly)
            Cqs.append(m2.matrices()[1])
        dCqv = (Cqs[0] - Cqs[1]) @ _full(nb, V) / (2.0 * fd_h)
        out["qc_fd"] = float(np.abs(dCqv + sol["Qc"]).max()
                             / max(np.abs(sol["Qc"]).max(), 1.0))

    if native:
        mn = fam.build(P0, V0)
        mn.sys.DoAssembly(0xFFFF)
        Pn, Vn = mn.state()
        stn = fam.generalised(Pn, Vn)
        out["native"] = relerr(fam.to_qdd(Pn, Vn, mn.native_acc()),
                               fam.ref_accel(stn))
    return out
