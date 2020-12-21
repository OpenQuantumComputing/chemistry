from energycalculator import *
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pickle
import matplotlib
matplotlib.use('Agg')

class EnergyCalculator(EnergyCalculatorBase):
    def get_geometry(self, dist):
        geometry = []
        geometry.append(("Be", [   0., 0., 0.]))
        geometry.append(("H" , [ dist, 0., 0.]))
        geometry.append(("H" , [-dist, 0., 0.]))
        return geometry

ecalc = EnergyCalculator()

backend = BasicAer.get_backend("statevector_simulator")
#backend = Aer.get_backend("qasm_simulator")
res=35
dists = np.linspace(1, 5.5, res)
exact_energies = []
ccsd_energies = []
quccsd_energies = []
quccsd_energies_best = {}
quccsd_parameters_best = {}

### 1) Hartree Fock-energies
i = 0
diagvalues = np.zeros((len(dists), 14))
for dist in dists:
    diagvalues[i,:] = ecalc.get_onebodyintegrals_HF(dist)
    i += 1
plt.figure(figsize=(20,10), dpi=400)
plt.plot(dists, diagvalues, 'x-')
plt.savefig('BeH2_HF.png')

#to reduce the number of qubits used, we freeze the core and remove two unoccupied orbitals
freeze_list = [0]
remove_list = []
#remove_list = [-4, -3]

### 2) energies obtianed by exact, ccsd, and quccsd methods
with open('BeH2_energies.txt', 'w') as file:
        file.write('d\texact\tquccsd\n')
for dist in dists:
    fstring=str(dist)+"\t"
    print("interatomic distance=", dist)
    print("exact energy", end=" ")
    exact_energy = ecalc.get_exact_energy(dist, backend)#, freeze_list = freeze_list, remove_list = remove_list)
    exact_energies.append(exact_energy)
    np.save('BeH2_linear_exact', exact_energies)
    print(exact_energy)
    fstring+=str(exact_energy)+"\t"
    print("quccsd energy", end=" ")
    quccsd_energy, quccsd_energies_best[dist], quccsd_parameters_best[dist] = ecalc.get_quccsd_energy(dist, backend, freeze_list = freeze_list, remove_list = remove_list)
    quccsd_energies.append(quccsd_energy)
    np.save('BeH2_linear_quccsd', quccsd_energies)
    fstring+=str(quccsd_energy)+"\t"
    with open('BeH2_linear_quccsd_Ebest.pkl', 'wb') as f:
        pickle.dump(quccsd_energies_best, f)
    with open('BeH2_linear_quccsd_paramsbest.pkl', 'wb') as f:
        pickle.dump(quccsd_parameters_best, f)
    print(quccsd_energy)
    with open('BeH2_energies.txt', 'a') as file:
            file.write(fstring+"\n")
    mode = "a"
    #print("ccsd energy", end=" ")
    #ccsd_energy = ecalc.get_ccsd_energy(dist)
    #ccsd_energies.append(ccsd_energy)
    #np.save('BeH2_linear_ccsd', ccsd_energies)
    #print(ccsd_energy)
    print()
    
plt.clf()
plt.plot(dists, exact_energies, 'o-', label="exact")
plt.plot(dists, quccsd_energies, 'x-',label="q-UCCSD")
#plt.plot(dists, ccsd_energies, 's-',label="CCSD")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.savefig('BeH2.png')

plt.clf()
#plt.plot(dists, np.array(exact_energies)-np.array(ccsd_energies), 's-',label="CCSD")
plt.plot(dists, np.array(quccsd_energies)-np.array(exact_energies), 'x-', label="q-UCCSD")
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel('Energy distance to exact solution')
plt.legend()
plt.savefig('BeH2diff.png')

pn = []
for key in quccsd_parameters_best:
    pn.append(LA.norm(quccsd_parameters_best[key],2))


plt.clf()
plt.plot(quccsd_parameters_best.keys(), pn,'x-')
plt.xlabel('Interatomic distance (Angstrom)')
plt.ylabel(r'$\|\theta_i\|_{l_2}$')



