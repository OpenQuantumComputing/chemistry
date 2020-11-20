from energycalculator import *
import matplotlib.pyplot as plt

class EnergyCalculator(EnergyCalculatorBase):
    def get_geometry(self, angle):
        R = 1.738
        angle *= np.pi/180 #conversion to radians
        a = R * np.cos(angle/2)
        b = R * np.sin(angle/2)
        geometry = []
        geometry.append(("H", [ a,  b, 0.]))
        geometry.append(("H", [-a,  b, 0.]))
        geometry.append(("H", [ a, -b, 0.]))
        geometry.append(("H", [-a, -b, 0.]))
        return geometry

ecalc = EnergyCalculator()

backend = BasicAer.get_backend("statevector_simulator")
#backend = Aer.get_backend("qasm_simulator")
res=15
angles = np.linspace(85, 90, res)
angles_sym = np.linspace(85,95, 2*res-1)
exact_energies = []
ccsd_energies = []
quccsd_energies = []

### 1) Hartree Fock-energies
i = 0
diagvalues = np.zeros((len(angles), 8))
for angle in angles:
    diagvalues[i,:] = ecalc.get_onebodyintegrals_HF(angle)
    i += 1
plt.figure(figsize=(20,10), dpi=400)
plt.plot(angles, diagvalues, 'x-')
plt.savefig('H4_HF.png')

### 2) energies obtianed by exact, ccsd, and quccsd methods
for angle in angles:
    print("angle=", angle)
    print("exact energy", end=" ")
    exact_energy = ecalc.get_exact_energy(angle, backend)
    exact_energies.append(exact_energy)
    print(exact_energy)
    print("quccsd energy", end=" ")
    quccsd_energy = ecalc.get_quccsd_energy(angle, backend)
    quccsd_energies.append(quccsd_energy)
    print(quccsd_energy)
    print("ccsd energy", end=" ")
    ccsd_energy = ecalc.get_ccsd_energy(angle)
    ccsd_energies.append(ccsd_energy)
    print(ccsd_energy)
    print()

### 3) use symmetry around 90 degrees
for i in range(res-1):
    exact_energies.append(exact_energies[res-i-2])
    quccsd_energies.append(quccsd_energies[res-i-2])
    ccsd_energies.append(ccsd_energies[res-i-2])
    
plt.clf()
plt.plot(angles_sym, exact_energies, 'o-', label="eigensolver")
plt.plot(angles_sym, quccsd_energies, 'x-',label="q-UCCSD")
plt.plot(angles_sym, ccsd_energies, 's-',label="CCSD")
plt.xlabel('Angle')
plt.ylabel('Energy')
plt.legend()
plt.savefig('H4.png')

plt.clf()
plt.plot(angles_sym, np.array(exact_energies)-np.array(ccsd_energies), 's-',label="CCSD")
plt.plot(angles_sym, np.array(exact_energies)-np.array(quccsd_energies), 'x-', label="quccsd")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy distance to exact solution')
plt.legend()
plt.savefig('H4diff.png')

