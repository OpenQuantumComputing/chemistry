from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
import numpy as np

# Short comment, only N2 works (ish) as intended right now, further work with H2O and H4 is required

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
        molecule = run_psi4(molecule, run_scf=False, run_ccsd=run_ccsd,
                            memory=8000, delete_input=True, delete_output=True)
        print("Finished with molecule calculations \n")
        if print_info:
            # Print out some results of calculation.
            print('\nAt bond length of {} angstrom, molecular nitrogen has:'.format(
                bond_length))
            print('CCSD energy of {} Hartree.'.format(molecule.ccsd_energy))
        ccsd_energies[i] = molecule.ccsd_energy
    return ccsd_energies

def get_ccsd_energies_H20(dist_list, print_info=False):
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
        molecule = run_psi4(molecule, run_scf=False, run_ccsd=run_ccsd,
                            memory=8000, delete_input=True, delete_output=True)
        print("Finished with molecule calculations \n")
        if print_info:
            # Print out some results of calculation.
            print('\nAt bond length of {} angstrom, molecular nitrogen has:'.format(
                bond_length))
            print('CCSD energy of {} Hartree.'.format(molecule.ccsd_energy))
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
    angle_list *= np.pi / 180 # Conversion to radians
    ccsd_energies = np.zeros(len(angle_list))
    for i in range(len(angle_list)):
        angle = angle_list[i]
        # Calculating the basis vectors
        a = R * np.cos(angle/2)
        b = R * np.sin(angle/2)
        geometry = [('H', (-a, -b, 0.)), ('H', (a, -b, 0.)), ('H', (a, b, 0.)), ('H', (-a, b, 0.))]
        molecule = MolecularData(geometry, basis, multiplicity, description=str(round(angle, 2)))
        print("Finished with initialization the geometry")
        # Run Psi4.
        molecule = run_psi4(molecule, run_scf=False, run_ccsd=run_ccsd,
                            memory=8000, delete_input=True, delete_output=True)
        print("Finished with molecule calculations \n")
        if print_info:
            # Print out some results of calculation.
            print('\nAt angle of {}, molecular nitrogen has:'.format(
                angle * 180 / np.pi ))
            print('CCSD energy of {} Hartree.'.format(molecule.ccsd_energy))
            print('Nuclear repulsion energy between protons is {} Hartree.'.format(
                molecule.nuclear_repulsion))
        ccsd_energies[i] = molecule.ccsd_energy
    return ccsd_energies