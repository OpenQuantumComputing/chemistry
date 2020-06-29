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
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, PyQuanteDriver
from qiskit.chemistry import FermionicOperator

"""
Short comment: The freeze_list is under control, makes sense intuitively, but the remove_list is harder.
The current computer can only deal with up to 12 Orbitals, and runs out of memory with more.
Removing the (degenerate) spin orbitals with the second highest energy seems to yield good results.
Further investigation is required concerning the removal of orbitals.

Configuring the necessary quantum chemistry libraries is hard to do in Windows, so if the simulations are ran on a 
Windows system, the driver "PyQuante" should be used. If on Linux, the driver PYSCF is included when installing qiskit,
and should be used. The speed of PYSCF is significantly much higher than PyQuante.
"""
def get_qubit_op_N2(distances, driver="pyquante", remove_list=[]):
    """
    :param distances: Array of distances in Ångstrøm that will be simulated
    :param driver: Specifies the chemistry driver
    :param remove_list: List of which oribtals to manually remove. [-2,-3] seems to work well.
    :return: Array of energies of the system corresponding to the distances array.
    """
    if driver=="pyquante":
        driver = PyQuanteDriver(atoms="N .0 .0 .0; N .0 .0 " + str(distances), units=UnitsType.ANGSTROM, charge=0)
    else:
        driver = driver = PySCFDriver(atom="N .0 .0 .0; N .0 .0 " + str(distances), unit=UnitsType.ANGSTROM,
                                      charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    freeze_list = [0, 1]
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    # This handles how the fermion mapping handles indices for the orbitals
    # Basically, it makes sure that to remove the two spin-degenerate orbitals
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    # Remove the frozen orbitals/electrons
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    # Remove the orbitals from remove_list, (i.e. orbitals with high energy)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
    # Reduce the #qubits needed using the underlying symmetries of the Hamiltonian
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = energy_shift + repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift

def get_exact_energies_N2(distances, driver="pyquante", remove_list=[]):
    """
    :param distances: Array of distances to calculate the exact energies for
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windows
    :return: Corresponding array of exact energies (exact for the quantum circuit perspective)
    """
    exact_energies = np.zeros(len(distances))
    for i in range(len(distances)):
        print(distances[i])
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_N2(distances[i], driver, remove_list)
        result = NumPyEigensolver(qubitOp).run()
        exact_energies[i] = np.real(result.eigenvalues) + shift
        print(exact_energies[i])
    return exact_energies

def get_VQE_energies_N2(distances, backend, driver="pyquante", remove_list=[], maxiter_optimizer=3):
    """
    :param distances: Array of distances to calculate the exact energies for
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windowsre
    :return: Corresponding array of the VQE energies obtained using q-UCCSD
    """
    VQE_energies = np.zeros(len(distances))
    for i in range(len(distances)):
        optimizer = SLSQP(maxiter=maxiter_optimizer)
        print("Distance: ", distances[i])
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_N2(distances[i], driver, remove_list)
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

def plot_accuracy(ccsd_energies, VQE_energies, exact_energies, distances, distances_ccsd_N2=None):
    """
    Plots the accuracy of the different numerical methods. Shifts the numerical energies such that they match
    the exact energy at the first distance. The value of the shift is indicated in the label
    :param ccsd_energies:
    :param VQE_energies:
    :param exact_energies:
    :param distances:
    :param distances_ccsd_N2: If a different distances list is used for the classical ccsd simulation.
    :return:
    """
    CHEM_ACCURACY = 1.6 * 10 ** -3
    rc('axes.formatter', useoffset=False)
    gs = gridspec.GridSpec(4, 1, width_ratios=[1], height_ratios=[1, 2, 3, 4])
    ax1 = plt.subplot(gs[:-1, :])
    ax2 = plt.subplot(gs[-1, :])
    ax1.plot(distances, exact_energies, "o-", alpha=0.4, label="Exact")
    ax1.plot(distances, VQE_energies + exact_energies[0] - VQE_energies[0], "o-", alpha=0.4,
             label=r"q-UCCSD, shift $= %0.3f$" % (exact_energies[0] - VQE_energies[0]))
    ax1.set_ylabel(r"Energy [Ha]")

    ax2.plot(distances, np.abs((VQE_energies + exact_energies[0] - VQE_energies[0]) - exact_energies), "o-",
             alpha=0.4, label=r"q-UCCSD")
    ax2.set_xlabel(r"$d$ [Å]")
    ax2.set_ylabel(r"$|\Delta E|$ [Ha]")
    ax2.hlines(CHEM_ACCURACY, np.amin(distances), np.amax(distances), linestyle="dashed", label="Chem acc.")
    if type(distances_ccsd_N2)==None:
        ax1.plot(distances, ccsd_energies + exact_energies[0] - ccsd_energies[0], "o-", alpha=0.4,
                 label=r"CCSD, shift $= %0.3f$" % (exact_energies[0] - ccsd_energies[0]))
        ax2.plot(distances, np.abs((ccsd_energies + exact_energies[0] - ccsd_energies[0]) - exact_energies), "o-",
                 alpha=0.4, label=r"CCSD")
    else:
        ax1.plot(distances_ccsd_N2, ccsd_energies + exact_energies[0] - ccsd_energies[0], "o-", alpha=0.4,
                 label=r"CCSD, shift $= %0.3f$" % (exact_energies[0] - ccsd_energies[0]))
        ax2.plot(distances_ccsd_N2, np.abs((ccsd_energies + exact_energies[0] - ccsd_energies[0]) - exact_energies[::2]), "o-",
                 alpha=0.4, label=r"CCSD")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()


