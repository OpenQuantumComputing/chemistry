from VVQE import *

backend = Aer.get_backend("qasm_simulator")
# Larger simulations
M = 4
N = 4
alpha_list_small = np.linspace(0, 0.4, M)
alpha_list_big = np.linspace(0.6, 1, M)
distance = 0.8
depth = 1

hamiltonian, shift = get_hamiltonian_H2(distance, "pyqsuante")
result = NumPyEigensolver(hamiltonian, k=20).run() # calculates k lowest eigengvalues
energy_spectrum = np.real(result.eigenvalues)
print(energy_spectrum)
E_0 = energy_spectrum[0]
print("Exact ground state energy: ", E_0)


init_params = np.ones(7)
'''
beta_list = np.linspace(1, 3, 5)
energy_matrix, std_matrix = find_effect_of_alpha(backend, beta_list, hamiltonian, 4, 2, init_params, depth=1, shots=1000, use_HF_initial=True,
                               var_form="UCCSD", entangler=linear,
                               cost_function=cost_function_beta, noise_model=None, method="COBYLA", k=5, save_results=True)

energy_matrix2, std_matrix2 =  find_effect_of_alpha(backend, alpha_list_small, hamiltonian, 4, 2, init_params, depth=1, shots=1000, use_HF_initial=False,
                               var_form="UCCSD", entangler=linear,
                               cost_function=cost_function_alpha, noise_model=None, method="COBYLA", k=80, save_results=True)


plot_effects_of_alpha(alpha_list_small, energy_matrix, std_matrix, "Var. form: UCCSD, Initial state: HF", E_0)
plot_effects_of_alpha(alpha_list_small, energy_matrix2, std_matrix2, "Var. form: UCCSD, Initial state: Comp. basis", E_0)

'''

# Reading data from previous simulations:
alpha_list = np.genfromtxt("data_VVQE/alpha_list_UCCSD_entangler_linear.csv", delimiter=",")
energy_matrix = np.genfromtxt("data_VVQE/energy_matrix_var_form_UCCSD_entangler_linear.csv", delimiter=",")
std_matrix = np.genfromtxt("data_VVQE/std_matrix_var_form_UCCSD_entangler_linear.csv", delimiter=",")
plot_effects_of_alpha(alpha_list, energy_matrix, std_matrix, r"Var. form: UCCSD, Initial state: HF, Cost function: $\beta$", E_0)


