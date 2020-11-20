from energycalculator import *
import matplotlib.pyplot as plt

class EnergyCalculator(EnergyCalculatorBase):
    def get_geometry(self, dist):
        geometry = []
        geometry.append(("Li", [   0., 0., 0.]))
        geometry.append(("H" , [ dist, 0., 0.]))
        return geometry

ecalc = EnergyCalculator()

backend = BasicAer.get_backend("statevector_simulator")
#backend = Aer.get_backend("qasm_simulator")
res=15
dists = np.linspace(0.5, 3.0, res)
exact_energies = []
ccsd_energies = []
quccsd_energies = []

### 1) Hartree Fock-energies
i = 0
diagvalues = np.zeros((len(dists), 12))
for dist in dists:
    diagvalues[i,:] = ecalc.get_onebodyintegrals_HF(dist)
    i += 1
plt.figure(figsize=(20,10), dpi=400)
plt.plot(dists, diagvalues, 'x-')
plt.savefig('LiH_HF.png')

#to reduce the number of qubits used, we freeze the core and remove two unoccupied orbitals
freeze_list = [0]
remove_list = [-3, -2]

### 2) energies obtianed by exact, ccsd, and quccsd methods
for dist in dists:
    print("interatomic distance=", dist)
    print("exact energy", end=" ")
    exact_energy = ecalc.get_exact_energy(dist, backend, freeze_list = freeze_list, remove_list = remove_list)
    exact_energies.append(exact_energy)
    print(exact_energy)
    print("quccsd energy", end=" ")
    quccsd_energy = ecalc.get_quccsd_energy(dist, backend, freeze_list = freeze_list, remove_list = remove_list)
    quccsd_energies.append(quccsd_energy)
    print(quccsd_energy)
    print("ccsd energy", end=" ")
    ccsd_energy = ecalc.get_ccsd_energy(dist)
    ccsd_energies.append(ccsd_energy)
    print(ccsd_energy)
    print()
    
plt.clf()
plt.plot(dists, exact_energies, 'o-', label="eigensolver")
plt.plot(dists, quccsd_energies, 'x-',label="q-UCCSD")
plt.plot(dists, ccsd_energies, 's-',label="CCSD")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.savefig('LiH.png')

plt.clf()
plt.plot(dists, np.array(exact_energies)-np.array(ccsd_energies), 's-',label="CCSD")
plt.plot(dists, np.array(exact_energies)-np.array(quccsd_energies), 'x-', label="quccsd")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy distance to exact solution')
plt.legend()
plt.savefig('LiHdiff.png')
