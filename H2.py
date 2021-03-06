from energycalculator import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class EnergyCalculator(EnergyCalculatorBase):
    def get_geometry(self, dist, string=False):
        if string:
            geometry  = "H " + str(dist) + " 0 0;"
            geometry += "H 0 0 0;"
            return geometry
        else:
            geometry = []
            geometry.append(("H", [ dist, 0., 0.]))
            geometry.append(("H", [   0., 0., 0.]))
            return geometry

ecalc = EnergyCalculator()

backend = BasicAer.get_backend("statevector_simulator")
#backend = Aer.get_backend("qasm_simulator")
res=15
dists = np.linspace(0.5, 3.0, res)
fci_energies_sto3g = []
fci_energies_ccpvdz = []
ccsd_energies_sto3g = []
ccsd_energies_ccpvdz = []
quccsd_energies_sto3g = []
#quccsd_energies_ccpvdz = []

### 1) Hartree Fock-energies
i = 0
diagvalues = np.zeros((len(dists), 4))
for dist in dists:
    diagvalues[i,:] = ecalc.get_onebodyintegrals_HF(dist)
    i += 1
plt.figure(figsize=(20,10), dpi=400)
plt.plot(dists, diagvalues, 'x-')
plt.savefig('H2_HF.png')

### 2) calculate energies
for dist in dists:
    print("interatomic distance=", dist)

    print("fci energy", end=" ")
    energy = ecalc.get_fci_energy(dist, basis = 'sto-3g')
    fci_energies_sto3g.append(energy)
    print("sto-3g=", energy, end=" ")
    energy = ecalc.get_fci_energy(dist, basis = 'ccpvdz')
    fci_energies_ccpvdz.append(energy)
    print("ccpvdz=", energy)
    #energy = ecalc.get_fci_energy(dist, basis = 'ccpvtz')
    #fci_energies_ccpvtz.append(energy)
    #print("ccpvtz=", energy)

    #print("exact energy", end=" ")
    #energy = ecalc.get_exact_energy(dist, backend)
    #exact_energies.append(energy)
    #print(energy)

    print("quccsd energy", end=" ")
    energy = ecalc.get_quccsd_energy(dist, backend, basis='sto-3g')
    quccsd_energies_sto3g.append(energy)
    print("sto-3g=", energy, end=" ")
    #energy = ecalc.get_quccsd_energy(dist, backend, basis= 'ccpvdz')
    #quccsd_energies_ccpvdz.append(energy)
    #print("ccpvdz=", energy)
    print()

    print("ccsd energy", end=" ")
    energy = ecalc.get_ccsd_energy(dist, basis = 'sto-3g')
    ccsd_energies_sto3g.append(energy)
    print("sto-3g=", energy, end=" ")
    energy = ecalc.get_ccsd_energy(dist, basis = 'cc-pVDZ')
    ccsd_energies_ccpvdz.append(energy)
    print("ccpvdz=", energy)

    print()
    
plt.clf()
plt.plot(dists, fci_energies_sto3g, 'x-', label="FCI sto-3g")
plt.plot(dists, fci_energies_ccpvdz, 'o:', label="FCI cc-pVDZ")
#plt.plot(dists, fci_energies_ccpvtz, '<-', label="FCI cc-pVTZ")
#plt.plot(dists, exact_energies, 'o-', label="eigensolver")
plt.plot(dists, quccsd_energies_sto3g, '<-',label="q-UCCSD sto-3g")
#plt.plot(dists, quccsd_energies_ccpvdz, 's:',label="q-UCCSD cc-pVDZ")
plt.plot(dists, ccsd_energies_sto3g, '*-',label="CCSD sto-3g")
plt.plot(dists, ccsd_energies_ccpvdz, '1:',label="CCSD cc-pVDZ")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.savefig('H2.png')

plt.clf()
plt.plot(dists, np.array(fci_energies_sto3g)-np.array(ccsd_energies_sto3g), 's-',label="CCSD")
plt.plot(dists, np.array(fci_energies_sto3g)-np.array(quccsd_energies_sto3g), 'x-', label="quccsd")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy distance to FCI solution')
plt.legend()
plt.savefig('H2diff_sto3g.png')

plt.clf()
plt.plot(dists, np.array(fci_energies_ccpvdz)-np.array(ccsd_energies_ccpvdz), 's-',label="CCSD")
#plt.plot(dists, np.array(fci_energies_ccpvdz)-np.array(quccsd_energies_ccpvdz), 'x-', label="quccsd")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy distance to FCI solution')
plt.legend()
plt.savefig('H2diff_ccpvdz.png')
