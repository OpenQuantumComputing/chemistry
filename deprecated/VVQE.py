from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, IBMQ
import numpy as np
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, PyQuanteDriver
from qiskit.chemistry import FermionicOperator
from qiskit import BasicAer, Aer, execute
from qiskit.aqua.algorithms import  NumPyEigensolver
from scipy.optimize import minimize
from qiskit.providers.aer.noise import NoiseModel
from qiskit.chemistry.components.initial_states import HartreeFock
from uccsd import *


import matplotlib.pyplot as plt
from matplotlib import rc
try:
    rc('text', usetex=True)
except:
    print("No LaTeX font available!")

newparams = {'figure.figsize': (10, 7), 'axes.grid': False,
             'lines.markersize': 10, 'lines.linewidth': 2,
             'font.size': 15, 'mathtext.fontset': 'stix',
             'font.family': 'STIXGeneral', 'figure.dpi': 200}
plt.rcParams.update(newparams)


def linear(qc, N, theta_vec):
    qc.cx(0, 1)
    for i in range(1, N - 1):
        qc.cx(i, i + 1)

def full(qc, N, theta_vec):
    for i in range(N - 1):
        for j in range(i + 1, N):
            qc.cx(i, j)

def particle_conservation_entanglement(qc, N, theta_vec):
    for i in range(N - 1):
        qc.cu3(theta = np.pi, lam =  -theta_vec[2*i] + np.pi,phi = -theta_vec[2*i],
               control_qubit=i, target_qubit = i+1)
        qc.cu3(theta=2*theta_vec[2*i + 1], lam=np.pi, phi=0, control_qubit=i+1, target_qubit=i)
        qc.cu3(theta = np.pi, lam = theta_vec[2*i] + np.pi,phi = theta_vec[2*i],
               control_qubit=i, target_qubit = i+1)

def HF_initial(N, m):
    initial_state = HartreeFock(
        N,
        m,
        qubit_mapping='parity',
        two_qubit_reduction= False
    )
    return initial_state.construct_circuit()

def create_VQE_circuit_RyRz(params, entangler,N ,depth=1, m=1, use_HF_initial=False):
    """
    Simple variational form described by qiskit
    :param params: Array of 2 * N * (depth + 1) elements,
    and 2 * (N - 1) * depth additional elements if the entangler is particle_conserv
    :param entangler: entangler function that takes a Quantum circuit and number of qubits as input
    :param N: #qubits
    :param depth:
    :param m: #electrons
    :param use_HF_inital:
    :return: quantum_circuit, quantum_register, classical_register of RyRz circuit
    """
    q = QuantumRegister(N)
    c = ClassicalRegister(N)
    qc = QuantumCircuit(q, c)
    if use_HF_initial:
        qc.combine(HF_initial(N, m))
    else:
        for j in range(m):
            qc.x(m)
    qc.barrier(q)
    counter = 0
    entangler_counter = 0
    for d in range(depth):
        for j in range(N):
            qc.ry(params[counter], j)
            counter += 1
        for j in range(N):
            qc.rz(params[counter], j)
            counter += 1
        entangler(qc, N, theta_vec=params[-1 - entangler_counter:-1 - entangler_counter - (2*N-1):-1])
        entangler_counter += 2*(N-1)
    for j in range(N):
        qc.ry(params[counter], j)
        counter += 1
    for j in range(N):
        qc.rz(params[counter], j)
        counter += 1
    return qc, q, c

def create_VQE_circuit_SC(params, entangler, N, depth=1, m=2, use_HF_initial=False):
    """
    Simple variational form described by Kandala et al. working well with IBM's real quantum computers
    (SC = Superconducting)
    :param params: Array of N * (3 * depth + 2) elements and 2* (N - 1) * depth
    additional elements if the entangler is particle_conserv
    :param entangler: entangler function that takes a Quantum circuit and number of qubits as input
    :param N: #qubits
    :param depth:
    :param m: #electrons
    :param use_HF_inital:
    :return: quantum_circuit, quantum_register, classical_register of SC circuit
    """
    q = QuantumRegister(N)
    c = ClassicalRegister(N)
    qc = QuantumCircuit(q,c)
    if use_HF_initial:
        qc.combine(HF_initial(N, m))
    else:
        for j in range(m):
            qc.x(m)
    qc.barrier(q)
    entangler_counter = 0
    counter = 0
    for i in range(N):
        qc.rx(params[2*i], i)
        qc.rz(params[2 * i + 1], i)
    counter += 2 * N
    for j in range(depth):
        entangler(qc, N, theta_vec=params[-1 - entangler_counter:-1 - entangler_counter - (2*N-1):-1])
        entangler_counter += 2*(N-1)
        qc.barrier()
        for i in range(N):
            theta_vec = params[counter:counter + 3]
            qc.rz(theta_vec[0], i)
            qc.rx(theta_vec[1], i)
            qc.rz(theta_vec[2], i)
            counter +=3
    return qc, q, c



def create_VQE_circuit_UCCSD(params, N, use_HF_initial=False):
    '''
    Credit to Teague Tomesh, the code is slightly modified here
    https://github.com/teaguetomesh/VQE/blob/master/Ansatz/UCCSD_Barkoutsos.py

    Creates quantum circuit equivalent to the Unitary Coupled Cluster Single Double Ansatz (UCCSD). Currently only works
    with H2 (i.e. 4 qubits)
    :param params: Array of 7 parameters
    :param N: #qubits
    :return: quantum_circuit, quantum_register, classical_register of UCCSD circuit
    '''
    if N is not 4:
        print('ERROR: UCCSD_4_Barkoutsos is currently implemented for 4 qubits only')
        sys.exit()

    # UCCSD ansatz for 4 qubits takes 7 different angles
    # 1 for the double excitation operator
    # 6 for the single excitation operator
    # Map the parameter indices to a dictionary here, indexed by p,q,r,s strings
    pdict = {'3210':0,'10':1,'20':2,'21':3,'30':4,'31':5,'32':6}

    # Initialize quantum register and circuit
    qreg = QuantumRegister(N, name='qreg')
    creg = ClassicalRegister(N, name='creg')
    circ  = QuantumCircuit(qreg, creg, name='UCCSD_4_Barkoutsos')

    if use_HF_initial:
        circ.combine(HF_initial(N, 2))
    else:
        for j in range(2):
            circ.x(2)
    circ.barrier(qreg)
    # enumerate all Nq > s > r > q > p >= 0 and apply Double Excitation Operator
    for s in range(N):
      for r in range(s):
        for q in range(r):
          for p in range(q):
            # For the 4 qubit case this function is called a single time
            srqp = str(s)+str(r)+str(q)+str(p)
            circ = DoubleExcitationOperator(circ, params[pdict[srqp]], s, r, q, p)

    # enumerate all Nq > r > p >= 0 and apply Single Excitation Operator
    for r in range(N):
      for p in range(r):
        rp = str(r)+str(p)
        SingleExcitationOperator(circ, params[pdict[rp]], r, p)

    return circ, qreg, creg

def get_hamiltonian_H2(distance, driver="pyquante"):
    if driver == "pyquante":
        driver = PyQuanteDriver(atoms="H .0 .0 .0; H .0 .0 " + str(distance), units=UnitsType.ANGSTROM, charge=0)
    else:
        driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(distance), unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto3g')
    molecule = driver.run()

    one_body = molecule.one_body_integrals
    two_body = molecule.two_body_integrals

    h = FermionicOperator(one_body, two_body)
    h = h.mapping("jordan_wigner")
    shift = molecule.nuclear_repulsion_energy
    return h,  shift

def cost_function_alpha(params, alpha, backend, hamiltonian, N, m, shots=1000, depth=1,
                        use_HF_initial=False, var_form="RyRz", entangler=linear, noise_model=None):
    if var_form=="RyRz":
        qc, q, c = create_VQE_circuit_RyRz(params, entangler,N ,depth, m, use_HF_initial)
    # Implementation of q-UCCSD is a work in progress
    elif var_form=="UCCSD":
        qc, q, c = create_VQE_circuit_UCCSD(params, N, use_HF_initial)
    elif var_form =="SC":
        qc, q, c = create_VQE_circuit_SC(params, entangler,N ,depth, m, use_HF_initial)
    else:
        print("WARNING: Could not find var_form instructions, using RyRz instead")
        qc, q, c = create_VQE_circuit_RyRz(params, entangler, N, depth)
    eval_circ_list = hamiltonian.construct_evaluation_circuit(wave_function=qc, statevector_mode=False, qr=q, cr=c)
    job = execute(eval_circ_list, backend, shots=shots)#, noise_model=noise_model)
    result = job.result()
    res = hamiltonian.evaluate_with_result(result=result, statevector_mode=False)
    mean = np.real(res[0])
    # res[1] is divided by sqrt(shots), thus the true variance must be scaled back
    std = np.real(res[1]) * np.sqrt(shots)
    return (1 - alpha) * mean + alpha * std


def cost_function_beta(params, beta, backend, hamiltonian, N, m, shots=1000, depth=1,
                        use_HF_initial=False, var_form="RyRz", entangler=linear, noise_model=None):
    if var_form=="RyRz":
        qc, q, c = create_VQE_circuit_RyRz(params, entangler,N ,depth, m, use_HF_initial)
    # Implementation of q-UCCSD is a work in progress
    elif var_form=="UCCSD":
        qc, q, c = create_VQE_circuit_UCCSD(params, N, use_HF_initial)
    elif var_form =="SC":
        qc, q, c = create_VQE_circuit_SC(params, entangler,N ,depth, m, use_HF_initial)
    else:
        print("WARNING: Could not find var_form instructions, using RyRz instead")
        qc, q, c = create_VQE_circuit_RyRz(params, entangler, N, depth)
    eval_circ_list = hamiltonian.construct_evaluation_circuit(wave_function=qc, statevector_mode=False, qr=q, cr=c)
    job = execute(eval_circ_list, backend, shots=shots)#, noise_model=noise_model)
    result = job.result()
    res = hamiltonian.evaluate_with_result(result=result, statevector_mode=False)
    mean = np.real(res[0])
    # res[1] is divided by sqrt(shots), thus the true variance must be scaled back
    std = np.real(res[1]) * np.sqrt(shots)
    return mean + beta * std


def find_optimal_params(backend, method, init_params, alpha, hamiltonian, N, m, shots, depth=1, use_HF_initial=False, var_form="RyRz",
                        entangler=linear, cost_function=cost_function_alpha, noise_model=None):
    optmize_result = minimize(cost_function, x0=init_params, method=method,
                              args=(alpha, backend, hamiltonian, N, m, shots, depth, use_HF_initial, var_form, entangler, noise_model),
                              options={"disp": False,"maxiter": 35})
    opt_params = optmize_result.x
    if var_form=="RyRz":
        qc, q, c = create_VQE_circuit_RyRz(opt_params, entangler, N, depth)
    elif var_form=="SC":
        qc, q, c = create_VQE_circuit_SC(opt_params, entangler, N, depth)
    elif var_form=="UCCSD":
        qc, q, c = create_VQE_circuit_UCCSD(opt_params, N, use_HF_initial)
    else:
        print("WARNING: Could not find variational form, using RyRz instead")
        qc, q, c = create_VQE_circuit_RyRz(opt_params, entangler, N, depth)
    eval_circ_list = hamiltonian.construct_evaluation_circuit(wave_function=qc, statevector_mode=False, qr=q, cr=c)
    job = execute(eval_circ_list, backend, shots=shots)
    result = job.result()
    res = hamiltonian.evaluate_with_result(result=result, statevector_mode=False)
    mean = np.real(res[0])
    error = np.real(res[1])
    return mean, error*np.sqrt(shots)



def simulate_variational_forms(backend, alpha_list,  hamiltonian, N, m, init_params, shots = 1000, depth=1, use_HF_initial = False,
                               var_form_entangler_list=[{"var_form":"RyRz", "entangler":linear}],
                               cost_function=cost_function_alpha, noise_model=None, method="COBYLA"):
    """
    :param backend:
    :param alpha_list:  Array of alphas to simulate. M values
    :param hamiltonian:
    :param init_params:
    :param depth:
    :param shots:
    :param var_form_entangler_list: List of dictionaries, with keys: "var_form" and "entangler"
    :param cost_function:
    :param noise_model:
    :param method:
    :return: error_mean_matrix: (N,M) matrix. Each element is the deviance between the deviance between the calculated
        energy for a unique pair of var_form and alpha and the closest exact eigenenergy.
        std_matrix : (N, M) matrix. Each element is the standard deviance for a unique pair of var_form and alpha
        resulting from calculating the energy.
        excited_states_matrix: (N, M) matrix. Specifies which eigenenergy that is closest to the calculated energy
        for a unique alpha, var_form pair.
    """
    M = len(alpha_list)
    k = len(var_form_entangler_list)
    result = NumPyEigensolver(hamiltonian, k=20).run() # calculates k lowest eigengvalues
    energy_spectrum = np.real(result.eigenvalues)
    print("Energy spectrum:")
    print(energy_spectrum)
    print()
    error_mean_matrix = np.zeros((k, M))
    std_matrix = np.zeros((k, M))
    excited_states_matrix = np.zeros((k, M), dtype=np.int)
    for i in range(k):
        var_form = var_form_entangler_list[i]["var_form"]
        print(var_form, " \t" , var_form_entangler_list[i]["entangler"].__name__)
        for j in range(len(alpha_list)):
            print("Simulation number: ", j + 1, ", \t alpha = ", alpha_list[j])
            mean, std = find_optimal_params(backend, method, init_params, alpha_list[j],  hamiltonian, N, m, shots, depth, use_HF_initial, var_form,
                        var_form_entangler_list[i]["entangler"], cost_function, noise_model)
            print("Mean: ", mean)
            print("STD: ", std)
            print()
            temp_rec = 1000
            index = 0
            for k in range(len(energy_spectrum)):
                if np.abs(energy_spectrum[k] - mean) < temp_rec:
                    temp_rec = np.abs(energy_spectrum[k] - mean)
                    index = k
            error_mean_matrix[i,j] = temp_rec
            std_matrix[i,j] = std
            excited_states_matrix[i,j] = index
    error_mean_matrix = np.abs(error_mean_matrix)
    return error_mean_matrix, std_matrix, excited_states_matrix



def plot_results_var_forms(alpha_list, distance, error_mean_matrix, std_matrix, excited_states_matrix,
                           var_form_entangler_list=[{"var_form":"RyRz", "entangler":linear}]):
    cmap = plt.get_cmap('gnuplot')
    M = len(alpha_list)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    unique_excited = np.unique(excited_states_matrix)
    colors = [cmap(i) for i in np.linspace(0, 1, np.amax(unique_excited) + 1)]
    for i in range(len(var_form_entangler_list)):
        label_string = "Var. form: " + var_form_entangler_list[i]["var_form"] + ", Entangler: " + var_form_entangler_list[i]["entangler"].__name__
        ax1.plot(alpha_list, std_matrix[i], label=label_string)
        ax1.set_xlabel(r"$\alpha$")
        ax1.set_ylabel(r"$\sigma$")
        ax1.set_title(r"$\sigma$ for $H_2$ with bond length %0.2f Å" % distance)
        ax1.legend()
        for j in range(M):
            ax2.scatter(alpha_list[j], error_mean_matrix[i, j], color=colors[int(excited_states_matrix[i, j])])
        ax2.plot(alpha_list, error_mean_matrix[i], alpha=0.5, linestyle="-", label=label_string)

    for j in range(len(unique_excited)):
        ax2.scatter(np.amin(alpha_list) - 10, np.amin(error_mean_matrix) + 10, color=colors[int(unique_excited[j])],
                    label=r'$k = $%0.0f' % unique_excited[j])
    ax2.set_xlim(min(np.amin(alpha_list) * 0.9, -0.01), np.amax(alpha_list) * 1.1)
    ax2.set_ylim(min(np.amin(alpha_list) * 0.9, -0.01), np.amax(error_mean_matrix) * 1.1)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$|\mu - E_k|$")
    ax2.set_title(r"Deviance between $\mu$ and the closest eigenvalue $E_k$")
    ax2.legend()
    plt.show()


def find_effect_of_alpha(backend, alpha_list,  hamiltonian, N, m, init_params, depth=1, shots=1000, use_HF_initial=False,
                               var_form="RyRz", entangler=linear,
                               cost_function=cost_function_alpha, noise_model=None, method="COBYLA", k=30, save_results=False):
    energy_matrix = np.zeros((len(alpha_list), k))
    std_matrix = np.zeros((len(alpha_list), k))
    for i in range(len(alpha_list)):
        for j in range(k):
            optmize_result = minimize(cost_function, x0=init_params, method=method,
                              args=(alpha_list[i], backend,  hamiltonian, N, m, shots, depth,
                                    use_HF_initial, var_form, entangler, noise_model),
                              options={"disp": False})#,"maxiter": 10}))
            opt_params = optmize_result.x
            if var_form == "RyRz":
                qc, q, c = create_VQE_circuit_RyRz(opt_params, entangler,hamiltonian.num_qubits, depth)
            elif var_form =="UCCSD":
                qc, q, c = create_VQE_circuit_UCCSD(opt_params, hamiltonian.num_qubits)
            elif var_form == "SC":
                qc, q, c = create_VQE_circuit_SC(opt_params, entangler,hamiltonian.num_qubits, depth)
            else:
                print("WARNING: Could not find var_form instructions, using RyRz instead")
                qc, q, c = create_VQE_circuit_RyRz(opt_params, entangler,hamiltonian.num_qubits, depth)
            eval_circ_list = hamiltonian.construct_evaluation_circuit(wave_function=qc, statevector_mode=False, qr=q, cr=c)
            job = execute(eval_circ_list, backend, shots=shots, noise_model=noise_model)
            result = job.result()
            res = hamiltonian.evaluate_with_result(result=result, statevector_mode=False)
            #print(result.get_counts())
            mean = np.real(res[0])
            error = np.real(res[1])
            std = np.sqrt(shots) * error
            print(j + 1, " alpha : ", alpha_list[i])
            print("Mean: ", mean)
            print("std: ", std)
            energy_matrix[i, j] = mean
            std_matrix[i, j] = std
    if save_results:
        universal_save_string = "var_form_" + var_form + "_entangler_" + entangler.__name__ + "_d_" \
                                + str(depth) + "_HF_initial_" + str(use_HF_initial) + \
                                "_cost_function_" + cost_function.__name__ + ".csv"
        energy_string =  "data_VVQE/energy_matrix_" + universal_save_string
        std_string = "data_VVQE/std_matrix_" + universal_save_string
        alpha_string = "data_VVQE/alpha_list_" + var_form + universal_save_string
        np.savetxt(energy_string, energy_matrix, delimiter=",")
        np.savetxt(std_string, std_matrix, delimiter=",")
        np.savetxt(alpha_string, alpha_list, delimiter=",")
    return energy_matrix, std_matrix


def plot_effects_of_alpha(alpha_list, energy_matrix, std_matrix, sup_title="", E_0=0):
    fig, ax = plt.subplots(len(alpha_list), 1, sharex=True, sharey=True)
    big, bx = plt.subplots(len(alpha_list), 1, sharex=True, sharey=True)
    for i in range(np.shape(energy_matrix)[0]):
        n, bins, patches = ax[i].hist(x=energy_matrix[i], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        ax[i].set_title(r"$\alpha = $ %0.2f" % alpha_list[i])
        ax[i].set_ylabel('Freq.')
        maxfreq = n.max()
        ax[i].axvline(np.mean(energy_matrix[i]), 0, maxfreq, color="k")
        ax[i].axvline(np.amin(energy_matrix[i]), 0, maxfreq, linestyle="--", color="r")
        if E_0 != 0:
            ax[i].axvline(E_0, 0, maxfreq, color="b", linestyle="--")
        n, bins, patches = bx[i].hist(x=std_matrix[i], bins="auto", color='#0504aa', alpha=0.7, rwidth=0.85)
        bx[i].set_ylabel('Frequency')
        maxfreq = n.max()
        bx[i].set_title(r"$\alpha = $ %0.2f" % alpha_list[i])
        bx[i].axvline(np.mean(std_matrix[i]), 0, maxfreq, color="k")

    ax[-1].set_xlabel(r'$\mu$')
    bx[-1].set_xlabel(r"$\sigma$")

    ax[-1].axvline(np.mean(energy_matrix[-1]), 0, maxfreq, color="k", label="Mean")
    bx[-1].axvline(np.mean(std_matrix[-1]), 0, maxfreq, color="k", label="Mean")
    ax[-1].axvline(np.amin(energy_matrix[-1]), 0, maxfreq, linestyle="--", color="r", label="Min")
    if E_0 != 0:
        ax[-1].axvline(E_0, 0, maxfreq, color="b", linestyle="--", label="Exact $E_0$")
    fig.legend()
    big.legend()
    fig.tight_layout()
    big.tight_layout()
    if sup_title!= "":
        fig.suptitle(sup_title)
        big.suptitle(sup_title)
        fig.subplots_adjust(top=0.85)
        big.subplots_adjust(top=0.85)
    plt.show()
