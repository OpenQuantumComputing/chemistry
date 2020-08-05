from HEA_VVQE import *

backend = Aer.get_backend("qasm_simulator")
# Larger simulations
M = 4
N = 4
alpha_list_small = np.genfromtxt("data_VVQE/alpha_list_SC_linear_small_alpha.csv", delimiter=",")
alpha_list_big = np.genfromtxt("data_VVQE/alpha_list2_SC_linear_big_alpha.csv", delimiter=",")

distance = 0.8
depth = 1

hamiltonian, shift = get_hamiltonian_H2(distance, "pyquante")
result = NumPyEigensolver(hamiltonian, k=20).run() # calculates k lowest eigengvalues
energy_spectrum = np.real(result.eigenvalues)
E_0 = energy_spectrum[0]
print("Actual ground state: ", E_0)
init_params = np.ones(N * (3 * depth + 2) + (N - 1) * depth)
energy_matrix, std_matrix = find_effect_of_alpha(backend, alpha_list_small, hamiltonian, init_params, depth=1, shots=1000,
                               var_form="SC", entangler=particle_conservation_entanglement,
                               cost_function=cost_function_alpha, noise_model=None, method="COBYLA", k=80)

np.savetxt("data_VVQE/energy_matrix_SC_cons_small_alpha.csv", energy_matrix, delimiter=",")
np.savetxt("data_VVQE/std_matrix_SC_cons_small_alpha.csv", std_matrix, delimiter=",")

energy_matrix3, std_matrix3 = find_effect_of_alpha(backend, alpha_list_big, hamiltonian, init_params, depth=1, shots=1000,
                               var_form="SC", entangler=particle_conservation_entanglement,
                               cost_function=cost_function_alpha, noise_model=None, method="COBYLA", k=80)

np.savetxt("data_VVQE/energy_matrix_SC_cons_big_alpha.csv", energy_matrix3, delimiter=",")
np.savetxt("data_VVQE/std_matrix_SC_cons_big_alpha.csv", std_matrix3, delimiter=",")



depth = 2
init_params = np.ones(N * (3 * depth + 2) + (N - 1) * depth)
energy_matrix2, std_matrix2 = find_effect_of_alpha(backend, alpha_list_small, hamiltonian, init_params, depth, shots=1000,
                               var_form="SC", entangler=particle_conservation_entanglement,
                               cost_function=cost_function_alpha, noise_model=None, method="COBYLA", k=80)

np.savetxt("data_VVQE/energy_matrix_SC_cons_small_alphad2.csv", energy_matrix2, delimiter=",")
np.savetxt("data_VVQE/std_matrix_SC_cons_small_alphad2.csv", std_matrix2, delimiter=",")

plot_effects_of_alpha(alpha_list_small, energy_matrix, std_matrix, "Var. form: SC, Entangler: Number conserving", E_0)
plot_effects_of_alpha(alpha_list_small, energy_matrix2, std_matrix2, "Var. form: SC, Entangler: Number conserving, d=2"), E_0
plot_effects_of_alpha(alpha_list_big, energy_matrix3, std_matrix3, "Var. form: SC, Entangler: Number conserving", E_0)


