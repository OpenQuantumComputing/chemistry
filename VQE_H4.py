import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
try:
    rc('text', usetex=True)
except:
    "Could not use LaTeX font"
import matplotlib.gridspec as gridspec

from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.aqua.components.optimizers import SLSQP, SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, PyQuanteDriver
from qiskit.chemistry import FermionicOperator

from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit import BasicAer


newparams = {'figure.figsize': (10, 7), 'axes.grid': False,
             'lines.markersize': 10, 'lines.linewidth': 2,
             'font.size': 15, 'mathtext.fontset': 'stix',
             'font.family': 'STIXGeneral', 'figure.dpi': 200}
plt.rcParams.update(newparams)



def get_qubit_op_H4(angle, driver="pyquante"):
    """
    :param distance: distance in Ångstrøm that will be simulated
    :param driver: Specifies the chemistry driver
    """
    R = 1.738
    angle *= np.pi/180
    a = np.round(np.sin(angle / 2) * R, 3)
    b = np.round(np.cos(angle / 2) * R, 3)
    if driver=="pyquante":
        driver = PyQuanteDriver(atoms="H " + str(a) + " " + str(b) + " .0; H " + str(-a) + " " + str(b) + " .0; H " + str(a) + " " + str(-b) + " .0; " + \
                                "H " + str(-a) + " " + str(-b) + " .0",
                                units=UnitsType.ANGSTROM, charge=0)
    else:
        driver = PySCFDriver(atom="H " + str(a) + " " + str(b) + " .0; H " + str(-a) + " " + str(b) + " .0"
                             , unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
    # Reduce the #qubits needed using the underlying symmetries of the Hamiltonian
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = molecule.nuclear_repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift


def get_exact_energies_H4(angles, driver="pyquante"):
    """
    :param angles: Array of angles to calculate the exact energies for. Input in degrees
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windows
    :return: Corresponding array of exact energies (exact for the quantum circuit perspective)
    """
    exact_energies = np.zeros(len(angles))
    for i in range(len(angles)):
        print(angles[i])
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_H4(angles[i], driver)
        result = NumPyEigensolver(qubitOp).run()
        exact_energies[i] = np.real(result.eigenvalues) + shift
        print(exact_energies[i])
    return exact_energies

def get_VQE_energies_H4(angles, backend, driver="pyquante", maxiter_optimizer=3):
    """
    :param angles: Array of distances to calculate the exact energies for
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windowsre
    :return: Corresponding array of the VQE energies obtained using q-UCCSD
    """
    VQE_energies = np.zeros(len(angles))
    for i in range(len(angles)):
        optimizer = SLSQP(maxiter=maxiter_optimizer)
        print("Angle: ", angles[i])
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_H4(angles[i], driver)
        print(num_particles)
        print(num_spin_orbitals)
        print(qubitOp.num_qubits)
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
        print("Initializing VQE")
        vqe = VQE(qubitOp, var_form, optimizer)
        print("Init complete")
        vqe_result = np.real(vqe.run(backend)['eigenvalue'] + shift)
        VQE_energies[i] = vqe_result
        vqe = None # Forcing the vqe object out of memory in order to avoid running out of memory
        qubitOp = None # Same as above
        print("Sucsess")
        print("Energy: ", VQE_energies[i])
    print("All energies have been calculated")
    return VQE_energies

backend = BasicAer.get_backend("statevector_simulator")

angles = np.linspace(85, 95, 11)
#get_exact_energies_H4(angles)
VQE_energies = get_VQE_energies_H4(angles, backend, maxiter_optimizer=10)

plt.plot(angles, VQE_energies, "o-", label="q-UCCSD")
plt.xlabel(r"$E$ [Ha]")
plt.ylabel(r"$\beta$ [deg]")
plt.legend()
plt.show()