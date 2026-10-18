import math
import pychrono as chrono

G, L, M, omega = 9.81, 1.0, 1.0, 3.0
theta = math.radians(35)

def read(label, getter):
    w = getter()                       # keep the wrench alive
    F = chrono.ChVector3d(w.force)     # copy the force out
    print(f"{label:<32} ({F.x:.6g}, {F.y:.6g}, {F.z:.6g})  |F| = {F.Length():.6g}")

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

read("R2 BEFORE assembly", link.GetReaction2)
system.DoAssembly(0xFFFF)
read("R1 AFTER DoAssembly", link.GetReaction1)
read("R2 AFTER DoAssembly", link.GetReaction2)
system.DoStepDynamics(1e-5)
read("R1 AFTER one 1e-5 s step", link.GetReaction1)
read("R2 AFTER one 1e-5 s step", link.GetReaction2)
print("gravity share only:", M * G * math.cos(theta))
print("exact |F|:         ", M * (G * math.cos(theta) + omega ** 2 * L))
