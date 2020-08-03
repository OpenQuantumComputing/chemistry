from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
import numpy as np


"""
See https://github.com/quantumlib/OpenFermion for how to install the openfermion library.
"""

def get_ccsd_energies_N2(dist_list, print_info=False):
    """
    :param dist_list: Array of distances (in Ångstrøm) used to calculate the CCSD energy of the N2 molecule
    :return: Array of CCSD energies corresponding to the distances array
    """
    # Set molecule parameters.
    basis = 'sto-3g'
    multiplicity = 1
    # Set calculation parameters.
    run_ccsd = 1
    ccsd_energies = np.zeros(len(dist_list))
    for i in range(len(dist_list)):
        bond_length = dist_list[i]
        geometry = [('N', (0., 0., 0.)), ('N', (0., 0., bond_length))]
        molecule = MolecularData(geometry, basis, multiplicity, description=str(round(bond_length, 2)))
        print("Finished with initialization the geometry")
        # Run Psi4.
        molecule = run_psi4(molecule, run_scf=True, run_ccsd=run_ccsd,
                            memory=8000, delete_input=True, delete_output=True)
        print("Finished with molecule calculations")
        if print_info:
            # Print out some results of calculation.
            print('At bond length of {} angstrom, molecular nitrogen has:'.format(np.round(
                bond_length, 3)))
            print('CCSD energy of {} Hartree.'.format(np.round(molecule.ccsd_energy,5)))
        ccsd_energies[i] = molecule.ccsd_energy
        print()
    return ccsd_energies

def get_ccsd_energies_H2O(dist_list, print_info=False):
    """
    :param dist_list: Array of distances (in Ångstrøm) used to calculate the CCSD energy of the H2O molecule
    :return: Array of CCSD energies corresponding to the distances array
    """
    # Set molecule parameters.
    alpha = 104.51 / 180 * np.pi # Internal angle in the water molecule
    # Set molecule parameters.
    basis = 'sto-3g'
    multiplicity = 1
    # Set calculation parameters.
    run_ccsd = 1
    ccsd_energies = np.zeros(len(dist_list))
    for i in range(len(dist_list)):
        bond_length = dist_list[i]
        a = np.round(np.sin(alpha/2) * bond_length, 3)
        b = np.round(np.cos(alpha/2) * bond_length, 3)
        geometry = [('H', (0., 0., 0.)), ('H', (2*a, 0., 0)), ('O', (a, b, 0))]
        molecule = MolecularData(geometry, basis, multiplicity, description=str(round(bond_length, 2)))
        print("Finished with initialization the geometry")
        # Run Psi4.
        molecule = run_psi4(molecule, run_scf=True, run_ccsd=run_ccsd,
                            memory=8000, delete_input=True, delete_output=True)
        print("Finished with molecule calculations")
        if print_info:
            # Print out some results of calculation.
            print('At bond length of {} angstrom, H2O has:'.format(np.round(
                bond_length, 3)))
            print('CCSD energy of {} Hartree.'.format(np.round(molecule.ccsd_energy, 5)))
        print()
        ccsd_energies[i] = molecule.ccsd_energy
    return ccsd_energies



def get_ccsd_energies_H4(angle_list, print_info=False, R=1.738):
    """
    :param angle_list: Input in degrees
    :param print_info:
    :return: Array of CCSD energies corresponding to the angle_list array
    """
    # Set molecule parameters.
    basis = 'sto-3g'
    multiplicity = 1
    # Set calculation parameters.
    run_ccsd = 1
    ccsd_energies = np.zeros(len(angle_list))
    for i in range(len(angle_list)):
        angle = angle_list[i] * np.pi / 180 # Conversion to radians
        # Calculating the basis vectors
        a = R * np.cos(angle/2)
        b = R * np.sin(angle/2)
        geometry = [('H', (-a, -b, 0.)), ('H', (a, -b, 0.)), ('H', (a, b, 0.)), ('H', (-a, b, 0.))]
        molecule = MolecularData(geometry, basis, multiplicity, description=str(round(angle, 2)))
        print("Finished with initialization the geometry")
        # Run Psi4.
        molecule = run_psi4(molecule, run_scf=True, run_ccsd=run_ccsd,
                            memory=8000, delete_input=True, delete_output=True)
        print("Finished with molecule calculations")
        if print_info:
            # Print out some results of calculation.
            print('At angle of {}, H4 has:'.format(np.round(
                angle * 180 / np.pi, 3)))
            print('CCSD energy of {} Hartree.'.format(np.round(molecule.ccsd_energy, 5)))
        ccsd_energies[i] = molecule.ccsd_energy
        print()
    return ccsd_energies