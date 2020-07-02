import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec

from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, PyQuanteDriver
from qiskit.chemistry import FermionicOperator



"""
Configuring the necessary quantum chemistry libraries is hard to do in Windows, so if the simulations are ran on a 
Windows system, the driver "PyQuante" should be used. Note that PyQuante requires python 3.6.
If on Linux, the driver PYSCF is included when installing qiskit, and should be used.
The speed of PYSCF is significantly much higher than PyQuante.
"""

""" 
The following three functions are for molecular nitrogen
"""
def get_qubit_op_N2(distance, driver="pyquante", remove_list=[]):
    """
    :param distance: Distance in Ångstrøm that will be simulated
    :param driver: Specifies the chemistry driver
    :param remove_list: List of which orbitals to manually remove. [-2,-3] seems to work well.
     Ideally, should remove none, but this requires more than 32 GB RAM.
    :return: Energy of the system corresponding to the distance.
    """
    if driver=="pyquante":
        driver = PyQuanteDriver(atoms="N .0 .0 .0; N .0 .0 " + str(distance), units=UnitsType.ANGSTROM, charge=0)
    else:
        driver = PySCFDriver(atom="N .0 .0 .0; N .0 .0 " + str(distance), unit=UnitsType.ANGSTROM,
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
    :param remove_list: List of what orbitals to manually remove, usually motivated by high energy oribtals.
    :return: Corresponding array of exact energies (exact for the quantum circuit perspective)
    """
    exact_energies = np.zeros(len(distances))
    for i in range(len(distances)):
        print("N_2 bond length = ", np.round(distances[i], 3))
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_N2(distances[i], driver, remove_list)
        result = NumPyEigensolver(qubitOp).run()
        exact_energies[i] = np.real(result.eigenvalues) + shift
        print("Exact energy = ", np.round(exact_energies[i], 5), "\n")
    return exact_energies

def get_VQE_energies_N2(distances, backend, driver="pyquante", remove_list=[], maxiter_optimizer=3):
    """
    :param distances: Array of distances to calculate the exact energies for
    :param backend: Backend to run qiskit simulation on
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windows
    :param remove_list: List of what orbitals to manually remove, usually motivated by high energy orbitals.
    :param maxiter_optimizer: How many iterations to use on the classical SLSQP optimizer.
    :return: Corresponding array of the VQE energies obtained using q-UCCSD
    """
    VQE_energies = np.zeros(len(distances))
    for i in range(len(distances)):
        optimizer = SLSQP(maxiter=maxiter_optimizer)
        print("N_2 bond length = ", np.round(distances[i], 3))
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_N2(distances[i], driver, remove_list)
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
        print("Success")
        print("VQE energy: ", np.round(VQE_energies[i], 5), "\n")
    print("All energies have been calculated")
    return VQE_energies


"""
The following three functions are for H_4
"""
def get_qubit_op_H4(angle, driver="pyquante"):
    """
    :param angle: Angle of between pairs of Hydrogen atoms, placed on a circle
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windows
    :return: Energy of the system corresponding to the angle
    """
    R = 1.738
    angle *= np.pi/180 #conversion to radians
    a = np.round(np.sin(angle / 2) * R, 3)
    b = np.round(np.cos(angle / 2) * R, 3)
    if driver=="pyquante":
        driver = PyQuanteDriver(atoms="H " + str(a) + " " + str(b) + " .0; H " + str(-a) + " " + str(b) + " .0; H " +\
                                      str(a) + " " + str(-b) + " .0; " + "H " + str(-a) + " " + str(-b) + " .0",
                                units=UnitsType.ANGSTROM, charge=0)
    else:
        driver = PySCFDriver(atom="H " + str(a) + " " + str(b) + " .0; H " + str(-a) + " " + str(b) + " .0; H " +\
                                      str(a) + " " + str(-b) + " .0; " + "H " + str(-a) + " " + str(-b) + " .0",
                             unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
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
        print("Internal angle H4 = ", np.round(angles[i], 3))
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_H4(angles[i], driver)
        result = NumPyEigensolver(qubitOp).run()
        exact_energies[i] = np.real(result.eigenvalues)  + shift
        print("Exact energy = ", np.round(exact_energies[i], 5), "\n")
    return exact_energies

def get_VQE_energies_H4(angles, backend, driver="pyquante", maxiter_optimizer=3):
    """
    :param angles: Array of distances to calculate the exact energies for
    :param backend: Backend to run qiskit simulation on
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windows
    :param maxiter_optimizer: How many iterations to use on the classical SLSQP optimizer.
    :return: Corresponding array of the VQE energies obtained using q-UCCSD
    """
    VQE_energies = np.zeros(len(angles))
    for i in range(len(angles)):
        optimizer = SLSQP(maxiter=maxiter_optimizer)
        print("Internal angle H4 = ", np.round(angles[i], 3))
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_H4(angles[i], driver)
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
        print("Success")
        print("VQE energy: ", np.round(VQE_energies[i], 5), "\n")
    print("All energies have been calculated")
    return VQE_energies



"""
The following three functions are for H2O
"""
def get_qubit_op_H2O(distance, driver="pyquante", remove_list=[]):
    """
    :param distance: Distance in Ångstrøm that will be simulated
    :param driver: Specifies the chemistry driver
    :param remove_list: List of which orbitals to manually remove. [-2,-3] seems to work well.
    :return: Energy of the system corresponding to the distance.
    """
    alpha = 104.51 * np.pi / 180
    a = np.round(np.sin(alpha / 2) * distance, 3)
    b = np.round(np.cos(alpha / 2) * distance, 3)
    if driver=="pyquante":
        driver = PyQuanteDriver(atoms="H .0 .0 .0; H .0 .0 " + str(2*a) + "; O " + str(a) + " " + str(b) + " .0",
                                units=UnitsType.ANGSTROM, charge=0)
    else:
        driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(2*a) + "; O " + str(a) + " " + str(b) + " .0", unit=UnitsType.ANGSTROM,
                         charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    freeze_list = [0]
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list) for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    # Remove the orbitals from remove_list, (i.e. orbitals with high energy)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = energy_shift + repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift



def get_exact_energies_H2O(distances, driver="pyquante", remove_list=[]):
    """
    :param distances: Array of distances to calculate the exact energies for
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windows
    :return: Corresponding array of exact energies (exact for the quantum circuit perspective)
    """
    exact_energies = np.zeros(len(distances))
    for i in range(len(distances)):
        print("Bond length OH = ", np.round(distances[i], 3))
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_H2O(distances[i], driver, remove_list)
        result = NumPyEigensolver(qubitOp).run()
        exact_energies[i] = np.real(result.eigenvalues) + shift
        print("Exact energy = ", np.round(exact_energies[i], 5), "\n")
    return exact_energies

def get_VQE_energies_H2O(distances, backend, driver="pyquante", remove_list=[], maxiter_optimizer=3):
    """
    :param distances: Array of distances to calculate the exact energies for
    :param backend: Backend to run qiskit simulation on
    :param driver: String of what chemistry driver to use. Use PYSCF for Linux, and pyquante for Windows
    :param remove_list: List of what orbitals to manually remove, usually motivated by high energy orbitals.
    :param maxiter_optimizer: How many iterations to use on the classical SLSQP optimizer.
    :return: Corresponding array of the VQE energies obtained using q-UCCSD
    """
    VQE_energies = np.zeros(len(distances))
    for i in range(len(distances)):
        optimizer = SLSQP(maxiter=maxiter_optimizer)
        print("Bond length OH = ", np.round(distances[i], 3))
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op_H2O(distances[i], driver, remove_list)
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
        vqe = None
        qubitOp = None
        print("Success")
        print("VQE energy = ", np.round(VQE_energies[i], 5), "\n")
    print("All energies have been calculated")
    return VQE_energies

def plot_accuracy(ccsd_energies, VQE_energies, exact_energies, x_variable, angle=False, x_variable_ccsd=None):
    """
    Plots the accuracy of the different numerical methods. Shifts the numerical energies such that they match
    the exact energy at the first distance. The value of the shift is indicated in the label
    :param ccsd_energies:
    :param VQE_energies:
    :param exact_energies:
    :param x_variable:
    :param x_variable_ccsd: If a different x-variable array is used for the classical ccsd simulation.
    :return:
    """
    CHEM_ACCURACY = 1.6 * 10 ** -3
    rc('axes.formatter', useoffset=False)
    gs = gridspec.GridSpec(4, 1, width_ratios=[1], height_ratios=[1, 2, 3, 4])
    ax1 = plt.subplot(gs[:-1, :])
    ax2 = plt.subplot(gs[-1, :])
    ax1.plot(x_variable, VQE_energies + exact_energies[0] - VQE_energies[0], "o-", alpha=0.4,
             label=r"q-UCCSD, shift $= %0.3f$" % (exact_energies[0] - VQE_energies[0]))
    ax1.set_ylabel(r"Energy [Ha]")

    ax2.plot(x_variable, np.abs((VQE_energies + exact_energies[0] - VQE_energies[0]) - exact_energies), "o",
             alpha=0.4, label=r"q-UCCSD")
    if angle:
        ax2.set_xlabel(r"$\beta$ [deg]")
    else:
       ax2.set_xlabel(r"$d$ [Å]")
    ax2.set_ylabel(r"$|\Delta E|$ [Ha]")
    ax2.hlines(CHEM_ACCURACY, np.amin(x_variable), np.amax(x_variable), linestyle="dashed", label="Chem acc.")
    if x_variable_ccsd == None:
        ax1.plot(x_variable, ccsd_energies + exact_energies[0] - ccsd_energies[0], "o-", alpha=0.4,
                 label=r"CCSD, shift $= %0.3f$" % (exact_energies[0] - ccsd_energies[0]))
        ax2.plot(x_variable, np.abs((ccsd_energies + exact_energies[0] - ccsd_energies[0]) - exact_energies), "o",
                 alpha=0.4, label=r"CCSD")
    else:
        ax1.plot(x_variable_ccsd, ccsd_energies + exact_energies[0] - ccsd_energies[0], "o-", alpha=0.4,
                 label=r"CCSD, shift $= %0.3f$" % (exact_energies[0] - ccsd_energies[0]))
        ax2.plot(x_variable_ccsd, np.abs((ccsd_energies + exact_energies[0] - ccsd_energies[0]) - exact_energies[::2]), "o",
                 alpha=0.4, label=r"CCSD")
    ax1.plot(x_variable, exact_energies, "o-", alpha=0.4, label="Exact")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()
