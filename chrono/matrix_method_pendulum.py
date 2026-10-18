"""Constrained acceleration and joint force recovered from Chrono's own
matrices (route described by D. Negrut, 30 Sep 2026; cf. chrono#846).
Single pendulum, PyChrono 10.0.0."""
import glob, math, os
import numpy as np
import pychrono as chrono

G, L, M = 9.81, 1.0, 1.0
theta = math.radians(35)
g = np.array([0.0, -G, 0.0])

def v3(v):
    return np.array([v.x, v.y, v.z])

def load_matrix(path):
    rows = [ln.split() for ln in open(path) if ln.strip()]
    data = np.array(rows, dtype=float)
    if data.shape[1] != 3:
        raise ValueError(f"{path}: expected 'row col value' triplets, got {data.shape[1]} columns")
    off = 1 if data[:, 0].min() >= 1 and data[:, 1].min() >= 1 else 0
    nr = int(data[:, 0].max()) - off + 1
    nc = int(data[:, 1].max()) - off + 1
    A = np.zeros((nr, nc))
    for r, c, val in data:
        A[int(r) - off, int(c) - off] += val
    return A, off

def build(omega):
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, -G, 0))
    ground = chrono.ChBody(); ground.SetFixed(True); system.Add(ground)
    bob = chrono.ChBody(); bob.SetMass(M)
    bob.SetInertiaXX(chrono.ChVector3d(1e-2, 1e-2, 1e-2))
    phat = chrono.ChVector3d(math.sin(theta), -math.cos(theta), 0)
    that = chrono.ChVector3d(math.cos(theta), math.sin(theta), 0)
    bob.SetPos(phat * L); bob.SetPosDt(that * (omega * L)); system.Add(bob)
    link = chrono.ChLinkDistance()
    link.Initialize(ground, bob, False, chrono.ChVector3d(0, 0, 0), phat * L, True)
    system.AddLink(link)
    return system, bob

rows = []
for i, omega in enumerate((0.5, 1.0, 2.0, 3.0)):
    system, bob = build(omega)
    system.DoAssembly(0xFFFF)

    # state read back AFTER assembly (assembly nudges v slightly)
    r = v3(bob.GetPos()); v = v3(bob.GetPosDt())
    d = np.linalg.norm(r); rh = r / d
    vperp2 = v @ v - (rh @ v) ** 2

    # closed-form constrained solution at this exact state
    mu = rh @ g + vperp2 / d
    a_ex = g - rh * mu
    F_ex = M * (a_ex - g)                    # joint force on the bob

    # reference guard: C, dC/dt, d2C/dt2 of the reference
    guard = max(abs(d - L), abs(rh @ v), abs(rh @ a_ex + vperp2 / d))

    # Chrono's matrices
    outdir = os.path.abspath(f"blocks/w{omega}")
    os.makedirs(outdir, exist_ok=True)
    for old in glob.glob(os.path.join(outdir, "*")):
        os.remove(old)
    system.WriteSystemMatrices(True, False, False, True, outdir + "/solve", True)
    files = sorted(glob.glob(os.path.join(outdir, "*")))
    fH = [p for p in files if p.endswith("_M.dat")]
    fC = [p for p in files if p.endswith("_Cq.dat")]
    if not fH or not fC:
        print("Could not find H/Cq files. Written:", files)
        raise SystemExit(1)
    H, offH = load_matrix(fH[0]); Cq, offC = load_matrix(fC[0])
    n = H.shape[0]
    if Cq.shape[1] < n:
        Cq = np.hstack([Cq, np.zeros((Cq.shape[0], n - Cq.shape[1]))])
    m = Cq.shape[0]

    if i == 0:
        print("files written:", [os.path.basename(p) for p in files])
        print("index base detected: H", offH, " Cq", offC)
        print("H shape", H.shape, " diag", np.round(np.diag(H), 6))
        print("Cq shape", Cq.shape, " row", np.round(Cq, 6))
    if n != 6 or m != 1:
        print(f"Unexpected sizes n={n}, m={m}; stopping so we can look at it.")
        raise SystemExit(1)

    # Qc = -(dCq/dt) v for the distance constraint at the bob's centre
    c_lin, c_rot = Cq[0, :3], Cq[0, 3:]
    k = c_lin @ rh                                   # signed scale of Cq along r-hat
    misalign = np.linalg.norm(c_lin - k * rh) + np.linalg.norm(c_rot)
    Qc = np.array([-k * vperp2 / d])

    f6 = np.concatenate([M * g, np.zeros(3)])        # gravity; no gyroscopic term (zero spin)

    # saddle-point solve:  H a + Cq^T lam = f ,  Cq a = Qc
    K = np.block([[H, Cq.T], [Cq, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.concatenate([f6, Qc]))
    a6, lam = sol[:n], sol[n:]
    F6 = -(Cq.T @ lam)                               # compare Cq^T lam, not lam

    err_a = np.linalg.norm(a6[:3] - a_ex)
    err_rot = np.linalg.norm(a6[3:])
    err_F = np.linalg.norm(F6[:3] - F_ex)
    err_chrono = np.linalg.norm(v3(bob.GetPosDt2()) - a_ex)

    # alternative: closed form substituted into Chrono's assembled equations
    a_ex6 = np.concatenate([a_ex, np.zeros(3)])
    lam_ex = np.linalg.lstsq(Cq.T, f6 - H @ a_ex6, rcond=None)[0]
    res_dyn = np.linalg.norm(H @ a_ex6 + Cq.T @ lam_ex - f6)
    res_con = np.linalg.norm(Cq @ a_ex6 - Qc)

    rows.append((omega, err_a, err_rot, err_F, err_chrono, res_dyn, res_con, guard, misalign))

print()
print(f"{'omega':>5} {'|a_mm-a_ex|':>12} {'|alpha_mm|':>11} {'|F_mm-F_ex|':>12} "
      f"{'|a_chr-a_ex|':>13} {'res_dyn':>9} {'res_con':>9} {'guard':>9} {'Cq_check':>9}")
for row in rows:
    print(f"{row[0]:>5.1f} {row[1]:>12.2e} {row[2]:>11.2e} {row[3]:>12.2e} "
          f"{row[4]:>13.6f} {row[5]:>9.1e} {row[6]:>9.1e} {row[7]:>9.1e} {row[8]:>9.1e}")
