from qc_energy_calculations import *
try:
    from classical_ccsd_calculations import*
except:
    "Something went wrong when importing the openfermion library. \n This is a required" \
    " library for this file to run"

from qiskit import BasicAer

backend = BasicAer.get_backend("statevector_simulator")

# Change these variables to run different parts of the simulations
save_calculations = True
redo_calculations = True
run_N2 = True
run_H4 = True
run_H2O = True

if redo_calculations:
    if run_N2:
        distances_N2 = np.linspace(1.0, 2.5, 15)
        exact_energies_N2 = get_exact_energies_N2(distances_N2, remove_list=[-2, -3], driver="PYSCF")
        VQE_energies_N2 = get_VQE_energies_N2(distances_N2, backend, remove_list=[-2, -3], driver="PYSCF")
        ccsd_energies_N2 = get_ccsd_energies_N2(distances_N2, print_info=True)
    if run_H4:
        angles_H4 = np.linspace(85, 95, 21)
        exact_energies_H4 = get_exact_energies_H4(angles_H4, driver="PYSCF")
        VQE_energies_H4 = get_VQE_energies_H4(angles_H4, backend, driver="PYSCF", maxiter_optimizer=10)
        ccsd_energies_H4 = get_ccsd_energies_H4(angles_H4, print_info=True)
    if run_H2O:
        distances_H2O = np.linspace(1.4, 2.4, 21)
        exact_energies_H2O = get_exact_energies_H2O(distances_H2O, driver="PYSCF")
        VQE_energies_H2O = get_VQE_energies_H2O(distances_H2O, backend, driver="PYSCF")
        ccsd_energies_H2O = get_ccsd_energies_H2O(distances_H2O, print_info=True)
    if save_calculations:
        np.savetxt("data/N2/distances_N2.csv", distances_N2, delimiter=",")
        np.savetxt("data/N2/exact_energies_N2.csv", exact_energies_N2, delimiter=",")
        np.savetxt("data/N2/VQE_energies_N2.csv", VQE_energies_N2, delimiter=",")
        np.savetxt("data/N2/ccsd_energies_N2.csv", ccsd_energies_N2, delimiter=",")

        np.savetxt("data/H4/angles_H4.csv", angles_H4, delimiter=",")
        np.savetxt("data/H4/exact_energies_H4.csv", exact_energies_H4, delimiter=",")
        np.savetxt("data/H4/VQE_energies_H4.csv", VQE_energies_H4, delimiter=",")
        np.savetxt("data/H4/ccsd_energies_H4.csv", ccsd_energies_H4, delimiter=",")

        np.savetxt("data/H2O/distances_H2O.csv", distances_H2O, delimiter=",")
        np.savetxt("data/H2O/exact_energies_H2O.csv", exact_energies_H2O, delimiter=",")
        np.savetxt("data/H2O/VQE_energies_H2O.csv", VQE_energies_H2O, delimiter=",")
        np.savetxt("data/H2O/ccsd_energies_H2O.csv", ccsd_energies_H2O, delimiter=",")
else:
    distances_N2 = np.genfromtxt("data/N2/distances_N2.csv", delimiter=",")
    exact_energies_N2 = np.genfromtxt("data/N2/exact_energies_N2.csv", delimiter=",")
    VQE_energies_N2 = np.genfromtxt("data/N2/VQE_energies_N2.csv", delimiter=",")
    ccsd_energies_N2 = np.genfromtxt("data/N2/ccsd_energies_N2.csv", delimiter=",")

    ccsd_energies_H4 = np.genfromtxt("data/H4/ccsd_energies_H4.csv", delimiter=",")
    angles_H4 = np.genfromtxt("data/H4/angles_H4.csv", delimiter=",")
    exact_energies_H4 = np.genfromtxt("data/H4/exact_energies_H4.csv", delimiter=",")
    VQE_energies_H4 = np.genfromtxt("data/H4/VQE_energies_H4.csv", delimiter=",")

    distances_H2O = np.genfromtxt("data/H2O/distances_H2O.csv", delimiter=",")
    exact_energies_H2O = np.genfromtxt("data/H2O/exact_energies_H2O.csv", delimiter=",")
    VQE_energies_H2O = np.genfromtxt("data/H2O/VQE_energies_H2O.csv", delimiter=",")
    ccsd_energies_H2O = np.genfromtxt("data/H2O/ccsd_energies_H2O.csv", delimiter=",")
