# This function requires both the open fermion library https://github.com/quantumlib/OpenFermion
# and the library Psi4 https://github.com/quantumlib/OpenFermion-Psi4
# Note that the Psi4 library prints a non-fatal error message at each iteration.

from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
import numpy as np



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
        print("Finished with initialization the geometry in the Psi4 module")
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
