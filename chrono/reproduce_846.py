"""Reproducer for projectchrono/chrono#846 (after D. Negrut, 30 Sep 2026).
Single pendulum, PyChrono 10.0.0: DoAssembly(0xFFFF) then GetPosDt2()
compared against the closed-form constrained acceleration."""
import math
import pychrono as chrono

G, L, M = 9.81, 1.0, 1.0
theta = math.radians(35)

def run(omega):
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
    system.DoAssembly(0xFFFF)
    g = chrono.ChVector3d(0, -G, 0)
    lam = (g ^ phat) + (omega * L) ** 2 / L
    a_exact = g - phat * lam
    a_chrono = bob.GetPosDt2()
    err = (a_chrono - a_exact).Length()
    return err, omega ** 2 * L, link.GetReaction2().force.Length(), M * abs(lam)

print(f"{'omega':>6} {'|a_chr - a_ex|':>15} {'omega^2 L':>10} {'ratio':>8} {'F_chrono':>10} {'F_exact':>10}")
for omega in (0.5, 1.0, 2.0, 3.0):
    err, w2l, f_chr, f_ex = run(omega)
    print(f"{omega:>6.1f} {err:>15.6f} {w2l:>10.4f} {err / w2l:>8.4f} {f_chr:>10.4f} {f_ex:>10.4f}")
