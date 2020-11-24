#NB! This is copied from Teague Tomesh
# https://github.com/teaguetomesh/VQE/blob/master/Ansatz/UCCSD_Barkoutsos.py
'''
Teague Tomesh - 4/10/2019
Implementation of the UCCSD ansatz for use in the VQE algorithm.
Based on the description given in Barkoutsos et al.
(https://arxiv.org/abs/1001.3855?context=physics.chem-ph)
'''
import sys
import math

PI = math.pi


def U_d(i, circ, s, r, q, p, dagger=False):
    '''
    See Double Excitation Operator circuit in Fig 1b. of Barkoutsos et al 2018
    Y in Fig 1b of Barkoutsos et al 2018 really means Rx(-pi/2)
    '''

    if dagger:
        angle = PI / 2
    else:
        angle = -PI / 2

    qr = circ.qregs[0]

    if i == 1:
        circ.h(qr[s])
        circ.h(qr[r])
        circ.rx(angle, qr[q])
        circ.h(qr[p])
    elif i == 2:
        circ.rx(angle, qr[s])
        circ.h(qr[r])
        circ.rx(angle, qr[q])
        circ.rx(angle, qr[p])
    elif i == 3:
        circ.h(qr[s])
        circ.rx(angle, qr[r])
        circ.rx(angle, qr[q])
        circ.rx(angle, qr[p])
    elif i == 4:
        circ.h(qr[s])
        circ.h(qr[r])
        circ.h(qr[q])
        circ.rx(angle, qr[p])
    elif i == 5:
        circ.rx(angle, qr[s])
        circ.h(qr[r])
        circ.h(qr[q])
        circ.h(qr[p])
    elif i == 6:
        circ.h(qr[s])
        circ.rx(angle, qr[r])
        circ.h(qr[q])
        circ.h(qr[p])
    elif i == 7:
        circ.rx(angle, qr[s])
        circ.rx(angle, qr[r])
        circ.rx(angle, qr[q])
        circ.h(qr[p])
    elif i == 8:
        circ.rx(angle, qr[s])
        circ.rx(angle, qr[r])
        circ.h(qr[q])
        circ.rx(angle, qr[p])

    return circ


def CNOTLadder(circ, controlStartIndex, controlStopIndex):
    '''
    Applies a ladder of CNOTs, as in the dashed-CNOT notation at bottom of
    Table A1 of Whitfield et al 2010
    Qubit indices increase from bottom to top
    '''

    qr = circ.qregs[0]

    if controlStopIndex > controlStartIndex:
        delta = 1
        index = controlStartIndex + 1
        controlStopIndex += 1
    else:
        delta = -1
        index = controlStartIndex

    while index is not controlStopIndex:
        circ.cx(qr[index], qr[index - 1])
        index += delta

    return circ


def DoubleExcitationOperator(circ, theta, s, r, q, p):
    # Prerequisite: s > r > q > p

    qr = circ.qregs[0]

    for i in range(1, 9):
        circ = U_d(i, circ, s, r, q, p, dagger=False)

        circ = CNOTLadder(circ, s, r)
        circ.cx(qr[r], qr[q])
        circ = CNOTLadder(circ, q, p)

        circ.rz(theta, qr[p])  # Rz(reg[s], Theta_p_q_r_s[p][q][r][s]);

        circ = CNOTLadder(circ, p, q)
        circ.cx(qr[r], qr[q])
        circ = CNOTLadder(circ, r, s)

        circ.barrier(qr)

        circ = U_d(i, circ, s, r, q, p, dagger=True)

    return circ


def SingleExcitationOperator(circ, theta, r, p):
    # Prerequisite: r > p
    # See Single Excitation Operator circuit in Fig 1a. of Barkoutsos et al 2018

    qr = circ.qregs[0]

    circ.barrier(qr)

    circ.rx(-PI / 2, qr[r])
    circ.h(qr[p])
    circ = CNOTLadder(circ, r, p)
    circ.rz(theta, qr[p])  # Rz(reg[q], Theta_p_q[p][q])
    circ = CNOTLadder(circ, p, r)

    circ.barrier(qr)

    circ.rx(PI / 2, qr[r])
    circ.h(qr[p])

    circ.h(qr[r])
    circ.rx(-PI / 2, qr[p])
    circ = CNOTLadder(circ, r, p)
    circ.rz(theta, qr[p])  # Rz(reg[q], Theta_p_q[p][q])
    circ = CNOTLadder(circ, p, r)

    circ.barrier(qr)

    circ.h(qr[r])
    circ.rx(PI / 2, qr[p])

    return circ

