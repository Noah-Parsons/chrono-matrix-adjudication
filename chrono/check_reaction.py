import math
import pychrono as chrono

G, L, M, omega = 9.81, 1.0, 1.0, 3.0
theta = math.radians(35)

def f(v):
    return f"({v.x:.6g}, {v.y:.6g}, {v.z:.6g})  |F| = {v.Length():.6g}"

def safe(label, fn):
    try:
        print(f"{label:<34}", fn())
    except Exception as e:
        print(f"{label:<34} unavailable ({type(e).__name__}: {e})")

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

safe("system step size (GetStep)", lambda: system.GetStep())
safe("R2 force BEFORE assembly", lambda: f(link.GetReaction2().force))
system.DoAssembly(0xFFFF)
safe("R1 force AFTER DoAssembly", lambda: f(link.GetReaction1().force))
safe("R2 force AFTER DoAssembly", lambda: f(link.GetReaction2().force))
system.DoStepDynamics(1e-5)
safe("R1 force AFTER one 1e-5 s step", lambda: f(link.GetReaction1().force))
safe("R2 force AFTER one 1e-5 s step", lambda: f(link.GetReaction2().force))
print("exact |F| at omega = 3:", M * (G * math.cos(theta) + omega ** 2 * L))
