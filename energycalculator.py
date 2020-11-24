from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, BasicAer

from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import PySCFDriver, Molecule
from qiskit.chemistry import FermionicOperator

from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4

import numpy as np

class EnergyCalculatorBase:
    
    optimizer = SLSQP(maxiter=5)

    def __init__(self):
        pass

    global g_energy, g_params

    def printvalues(self, count, paramset,means,estimator):
        global g_energy, g_params
        g_energy[count] = means
        g_params[count] = paramset
        #print(count, paramset, means, estimator )

    def get_molecule(self, dist):
        multiplicity = 1
        geometry = self.get_geometry(dist)
        molecule = Molecule(geometry=geometry, multiplicity=multiplicity)
        return molecule

    def get_onebodyintegrals_HF(self, dist):
        molecule = self.get_molecule(dist)
        driver = PySCFDriver(molecule=molecule)
        qmolecule = driver.run()
        return np.diag(qmolecule.one_body_integrals)

    def get_qubit_op(self, dist, remove_list=[], freeze_list=[]):
        molecule = self.get_molecule(dist)
        driver = PySCFDriver(molecule=molecule)
        qmolecule = driver.run()
        num_particles = qmolecule.num_alpha + qmolecule.num_beta
        num_spin_orbitals = qmolecule.num_orbitals * 2
        ferOp = FermionicOperator(h1=qmolecule.one_body_integrals, h2=qmolecule.two_body_integrals)
        if len(remove_list)>0:
            remove_list = [x % qmolecule.num_orbitals for x in remove_list]
        if len(freeze_list)>0:
            freeze_list = [x % qmolecule.num_orbitals for x in freeze_list]
        if len(remove_list)>0:
            remove_list = [x - len(freeze_list) for x in remove_list]
            remove_list += [x + qmolecule.num_orbitals - len(freeze_list)  for x in remove_list]
        if len(freeze_list)>0:
            freeze_list += [x + qmolecule.num_orbitals for x in freeze_list]
            ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
            num_spin_orbitals -= len(freeze_list)
            num_particles -= len(freeze_list)
        else:
            energy_shift = 0.
        if len(remove_list)>0:
            ferOp = ferOp.fermion_mode_elimination(remove_list)
            num_spin_orbitals -= len(remove_list)
        qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
        qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
        shift = energy_shift + qmolecule.nuclear_repulsion_energy
        return qubitOp, num_particles, num_spin_orbitals, shift

    def get_ccsd_energy(self, dist):
        basis = 'sto-3g'
        multiplicity = 1

        geometry = self.get_geometry(dist)
        moleculedata = MolecularData(geometry, basis, multiplicity, description=str(round(dist, 2)))
        molecule = run_psi4(moleculedata, run_scf=True, run_ccsd=True, delete_input=True, delete_output=True)
        return molecule.ccsd_energy

    def get_exact_energy(self, dist, backend, remove_list=[], freeze_list=[]):
        qubitOp, num_particles, num_spin_orbitals, shift = self.get_qubit_op(dist, remove_list=remove_list, freeze_list=freeze_list)

        result = NumPyEigensolver(qubitOp).run()
        exact_energy = np.real(result.eigenvalues) + shift
        return exact_energy[0]

    def get_quccsd_energy(self, dist, backend, remove_list=[], freeze_list=[]):
        qubitOp, num_particles, num_spin_orbitals, shift = self.get_qubit_op(dist, remove_list=remove_list, freeze_list=freeze_list)

        initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_mapping='parity') 
        var_form = UCCSD(num_orbitals=num_spin_orbitals, num_particles=num_particles, initial_state=initial_state, qubit_mapping='parity')

        global g_energy, g_params
        g_energy = {}
        g_params = {}

        vqe = VQE(qubitOp, var_form, self.optimizer, callback=self.printvalues)
        quccsd_energy = np.real(vqe.run(backend)['eigenvalue'] + shift)
        return quccsd_energy 
