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
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel

from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4

from pyscf import gto, scf

global g_energy, g_params

def printvalues(count, paramset,means,estimator):
    global g_energy, g_params
    g_energy[count] = means
    g_params[count] = paramset
    #print(count, paramset, means, estimator )

def get_geometry(angle):
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

def get_molecule(angle):
    multiplicity = 1
    geometry = get_geometry(angle)
    molecule = Molecule(geometry=geometry, multiplicity=multiplicity)
    return molecule

def get_onebodyintegrals_HF(angle):
    molecule = get_molecule(angle)
    driver = PySCFDriver(molecule=molecule)
    qmolecule = driver.run()
    return np.diag(qmolecule.one_body_integrals)

def get_qubit_op(angle):
    molecule = get_molecule(angle)
    driver = PySCFDriver(molecule=molecule)
    qmolecule = driver.run()
    num_particles = qmolecule.num_alpha + qmolecule.num_beta
    num_spin_orbitals = qmolecule.num_orbitals * 2
    ferOp = FermionicOperator(h1=qmolecule.one_body_integrals, h2=qmolecule.two_body_integrals)
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = qmolecule.nuclear_repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift

def get_ccsd_energy(angle):
    basis = 'sto-3g'
    multiplicity = 1

    geometry = get_geometry(angle)
    moleculedata = MolecularData(geometry, basis, multiplicity, description=str(round(angle, 2)))
    molecule = run_psi4(moleculedata, run_scf=True, run_ccsd=True, delete_input=True, delete_output=True)
    return molecule.ccsd_energy

def get_quccsd_energy(angle, exact=False):
    qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op(angle)

    exact_energy=None
    if exact:
        result = NumPyEigensolver(qubitOp).run()
        exact_energy = np.real(result.eigenvalues) + shift

    initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_mapping='parity') 
    var_form = UCCSD(num_orbitals=num_spin_orbitals, num_particles=num_particles, initial_state=initial_state, qubit_mapping='parity')

    global g_energy, g_params
    g_energy = {}
    g_params = {}

    vqe = VQE(qubitOp, var_form, optimizer, callback=printvalues)
    quccsd_energy = np.real(vqe.run(backend)['eigenvalue'] + shift)
    return quccsd_energy, exact_energy


backend = BasicAer.get_backend("statevector_simulator")
#backend = Aer.get_backend("qasm_simulator")
#angles = np.array([85, 90])
res=10
angles = np.linspace(85, 90, res)
angles_sym = np.linspace(85,95, 2*res-1)
exact_energies = []
ccsd_energies = []
quccsd_energies = []
optimizer = SLSQP(maxiter=5)
#optimizer = COBYLA(maxiter=250)

### 1) Hartree Fock-energies
i = 0
diagvalues = np.zeros((len(angles), 8))
for angle in angles:
    diagvalues[i,:] = get_onebodyintegrals_HF(angle)
    i += 1
plt.plot(angles, diagvalues, 'x-')
plt.savefig('HF.png')

### 2) energies obtianed by exact, ccsd, and quccsd methods
for angle in angles:
    print("quccsd and exact energy at", angle, "degrees")
    quccsd_energy, exact_energy = get_quccsd_energy(angle, True)
    print("ccsd at", angle, "degrees")
    ccsd_energy = get_ccsd_energy(angle)
    exact_energies.append(exact_energy)
    ccsd_energies.append(ccsd_energy)
    quccsd_energies.append(quccsd_energy)
    
### 3) use symmetry around 90 degrees
for i in range(res-1):
    exact_energies.append(exact_energies[res-i-2])
    quccsd_energies.append(quccsd_energies[res-i-2])
    ccsd_energies.append(ccsd_energies[res-i-2])

plt.clf()
plt.plot(angles_sym, exact_energies, 'x-', label="eigensolver")
plt.plot(angles_sym, quccsd_energies, 'x-',label="q-UCCSD")
plt.plot(angles_sym, ccsd_energies, 'x-',label="CCSD")
plt.xlabel('Angle')
plt.ylabel('Energy')
plt.legend()
plt.savefig('H4.png')

