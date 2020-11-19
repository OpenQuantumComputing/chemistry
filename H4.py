from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from qiskit import Aer, execute
backend = Aer.get_backend("qasm_simulator")

from qiskit.aqua.components.optimizers import COBYLA

from qiskit.aqua.algorithms import VQE, NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit import IBMQ, BasicAer, Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel

from classical_ccsd_calculations import *

global g_energy, g_params
def printvalues(count, paramset,means,estimator):
    global g_energy, g_params
    g_energy[count] = means
    g_params[count] = paramset
    #print(count, paramset, means, estimator )

def get_qubit_op_HF(angle):
    R = 1.738
    angle *= np.pi/180 #conversion to radians
    a = np.round(np.sin(angle / 2) * R, 3)
    b = np.round(np.cos(angle / 2) * R, 3)
    driver = PySCFDriver(atom="H " + str(a) + " " + str(b) + " .0; H " + str(-a) + " " + str(b) + " .0; H " +\
                                      str(a) + " " + str(-b) + " .0; " + "H " + str(-a) + " " + str(-b) + " .0",
                             unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    return np.diag(molecule.one_body_integrals)

def get_qubit_op(angle):
    R = 1.738
    angle *= np.pi/180 #conversion to radians
    a = np.round(np.sin(angle / 2) * R, 3)
    b = np.round(np.cos(angle / 2) * R, 3)
    driver = PySCFDriver(atom="H " + str(a) + " " + str(b) + " .0; H " + str(-a) + " " + str(b) + " .0; H " +\
                                      str(a) + " " + str(-b) + " .0; " + "H " + str(-a) + " " + str(-b) + " .0",
                             unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = molecule.nuclear_repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift


backend = BasicAer.get_backend("statevector_simulator")
#backend = Aer.get_backend("qasm_simulator")
#angles = np.array([85, 90])
res=10
angles = np.linspace(85, 90, res)
angles_sym = np.linspace(85,95, 2*res-1)
exact_energies = []
quccsdvqe_energies = []
optimizer = SLSQP(maxiter=5)
#optimizer = COBYLA(maxiter=250)

diagvalues = np.zeros((len(angles), 8))

i = 0
for angle in angles:
    diagvalues[i,:] = get_qubit_op_HF(angle)
    i += 1

print(diagvalues)

plt.plot(angles, diagvalues)
plt.savefig('HF.png')

for angle in angles:
    qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op(angle)
    result = NumPyEigensolver(qubitOp).run()
    exact_energies.append(np.real(result.eigenvalues) + shift)
    initial_state = HartreeFock(
        num_spin_orbitals,
        num_particles,
        qubit_mapping='parity'
    ) 
    var_form = UCCSD(
        num_orbitals=num_spin_orbitals,
        num_particles=num_particles,
        initial_state=initial_state,
        qubit_mapping='parity'
    )

    g_energy = {}
    g_params = {}

    vqe = VQE(qubitOp, var_form, optimizer, callback=printvalues)
    vqe_result = np.real(vqe.run(backend)['eigenvalue'] + shift)
    quccsdvqe_energies.append(vqe_result)
    print("Angle=", np.round(angle, 2), "VQE Result:", vqe_result, "Exact Energy:", exact_energies[-1])
    ind = min(g_energy, key=g_energy.get)
    print(g_energy[ind]+shift, g_params[ind])
    print(list(g_energy.items())[-1]+shift, list(g_params.items())[-1])
    print()
    print()
    print()
    
print("All energies have been calculated")

ccsd_energies = list(get_ccsd_energies_H4(angles))

for i in range(res-1):
    exact_energies.append(exact_energies[res-i-2])
    quccsdvqe_energies.append(quccsdvqe_energies[res-i-2])
    ccsd_energies.append(ccsd_energies[res-i-2])

plt.clf()
plt.plot(angles_sym, exact_energies, 'x-', label="Exact Energy")
plt.plot(angles_sym, quccsdvqe_energies, 'x-',label="VQE Energy")
plt.plot(angles_sym, ccsd_energies, 'x-',label="CCSD Energy")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.savefig('H4.png')

