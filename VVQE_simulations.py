from HEA_VVQE import *

backend = Aer.get_backend("qasm_simulator")
# Larger simulations
M = 4

alpha_list_small = np.linspace(0.0, 0.4, M)
np.savetxt("data_VVQE/alpha_list_SC_linear_small_alpha.csv", alpha_list_small, delimiter=",")

alpha_list_large = np.linspace(0.6, 1, M)
np.savetxt("data_VVQE/alpha_list2_SC_linear_big_alpha.csv", alpha_list_large, delimiter=",")

var_form = "SC"
distance = 0.8



energy_matrix1, std_matrix1, shift1 = find_effect_of_alpha(backend, var_form, alpha_list_small, distance, shots=1000,
                                                           driver="jasf", entangler=linear_entangler, k=80)
np.savetxt("data_VVQE/energy_matrix_SC_linear_small_alpha.csv", energy_matrix1, delimiter=",")
np.savetxt("data_VVQE/std_matrix_SC_linear_small_alpha.csv", std_matrix1, delimiter=",")




energy_matrix2, std_matrix2, shift2 = find_effect_of_alpha(backend, var_form, alpha_list_large, distance,
                                                           shots=1000, driver="jasf", k=80,entangler= linear_entangler)
np.savetxt("data_VVQE/energy_matrix_SC_linear_big_alpha.csv", energy_matrix2, delimiter=",")
np.savetxt("data_VVQE/std_matrix_SC_linear_big_alpha.csv", std_matrix2, delimiter=",")




energy_matrix3, std_matrix3, shift3 = find_effect_of_alpha(backend, var_form, alpha_list_small, distance,
                                                           shots=1000, driver="jasf", k=80, entangler= full_entangler)
np.savetxt("data_VVQE/energy_matrix_SC_full_small_alpha.csv", energy_matrix3, delimiter=",")
np.savetxt("data_VVQE/std_matrix_SC_full_small_alpha.csv", std_matrix3, delimiter=",")





energy_matrix4, std_matrix4, shift4 = energy_matrix3, std_matrix3, shift3 = find_effect_of_alpha(backend, var_form, alpha_list_large, distance,
                                                           shots=1000, driver="jasf", k=80, entangler= full_entangler)
np.savetxt("data_VVQE/energy_matrix_SC_full_big_alpha.csv", energy_matrix4, delimiter=",")
np.savetxt("data_VVQE/std_matrix_SC_full_big_alpha.csv", std_matrix4, delimiter=",")




plot_effects_of_alpha(alpha_list_small, energy_matrix1, std_matrix1, SHIFT)
plot_effects_of_alpha(alpha_list_large, energy_matrix2, std_matrix2, SHIFT)
plot_effects_of_alpha(alpha_list_small, energy_matrix3, std_matrix3, SHIFT)
plot_effects_of_alpha(alpha_list_large, energy_matrix4, std_matrix4, SHIFT)

