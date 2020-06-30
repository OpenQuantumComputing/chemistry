from quantum_chem_VQE_N2 import *
from qiskit import BasicAer


backend = BasicAer.get_backend("statevector_simulator")
driver = "PYSCF"
redo_caculations = True
save_calcuations = False

if redo_caculations:
    distances_N2 = np.arange(1.0, 2.7, 0.1)
    VQE_energies_N2 = get_VQE_energies_N2(distances_N2, backend, driver, remove_list=[-2,-3])
    exact_energies_N2 = get_exact_energies_N2(distances_N2, driver, remove_list=[-2,-3])
    distances_ccsd_N2 = np.arange(1.0, 2.7, 0.2)
    try:
        from quantum_chem_psi4_N2 import get_ccsd_energies_N2
        ccsd_energies_N2 = get_ccsd_energies_N2(distances_ccsd_N2, print_info=True)
    except:
        "Could not import the openfermion library"
        distances_ccsd_N2 = np.genfromtxt("data/distances_ccsd_N2.csv", delimiter=",")
        ccsd_energies_N2 = np.genfromtxt("data/distances_ccsd_N2.csv", delimiter=",")
    if save_calcuations:
        np.savetxt("data/distances_N2.csv", distances_N2, delimiter=",")
        np.savetxt("data/exact_energies_N2.csv", exact_energies_N2, delimiter=",")
        np.savetxt("data/VQE_energies_N2.csv", VQE_energies_N2, delimiter=",")
        np.savetxt("data/distances_ccsd_N2.csv", distances_ccsd_N2, delimiter=",")
        np.savetxt("data/ccsd_energies_N2.csv", ccsd_energies_N2, delimiter=",")
else:
    distances_N2 = np.genfromtxt("data/distances_N2.csv", delimiter=",")
    exact_energies_N2 = np.genfromtxt("data/exact_energies_N2.csv", delimiter=",")
    VQE_energies_N2 = np.genfromtxt("data/VQE_energies_N2.csv", delimiter=",")
    distances_ccsd_N2 = np.genfromtxt("data/distances_ccsd_N2.csv", delimiter=",")
    ccsd_energies_N2 = np.genfromtxt("data/energies_ccsd_N2.csv", delimiter=",")

plot_accuracy(ccsd_energies_N2, VQE_energies_N2, exact_energies_N2, distances_N2, distances_ccsd_N2)
