import numpy as np
import matplotlib.pyplot as plt

import fastmat as fm
import fastmat.algorithms as fma

# Problem dimensions
compression_factor = 5
N = 256  # Problem size

# Number of observations (specified over compression factor)
M = int(N / compression_factor)
k = 3  # sparsity level
noise_power_db = -3

# Ground truth of scenario
ground_truth = np.zeros((N,))
ground_truth[np.random.choice(N, k, replace=False)] = np.random.randn(k)


# Set up the linear signal model and reconstruction method,
# consisting of Measurement Matrix `Phi` and Signal Base `Dict`
Phi = fm.Matrix(np.random.randn(M, N))
Dict = fm.Hadamard(8)

A = Phi * Dict
alg_omp = fma.OMP(A)
alg_fista = fma.FISTA(A)
alg_stela = fma.STELA(A)

# Now determine the actual (real-world) signal and its observation
# according to the specified Measurement matrix and plot the signals
# also allow for noise
x_clean = Dict * ground_truth
x = x_clean + 10 ** (noise_power_db / 10.0) * np.sqrt(1 / N) * (
    np.linalg.norm(ground_truth) * np.random.randn(*ground_truth.shape)
)
b = Phi * x

# Now reconstruct the original ground truth using
# * Orthogonal Matching Pursuit (OMP)
# * Fast Iterative Shrinkage Thresholding Algorithm (FISTA)
# * Soft-Thresholding with simplified Exact Line search Algorithm (STELA)
numLambda = 1.8
numSteps = 600
y_omp = alg_omp.process(b, numMaxSteps=k)
y_fista = alg_fista.process(b, numMaxSteps=numSteps, numLambda=numLambda)
y_stela = alg_stela.process(b, numMaxSteps=numSteps, numLambda=numLambda)

# Setup a simple phase transition diagram for OMP:
trials = 50
M_phase_transition = np.arange(k, N)
true_defect_positions = np.argsort(np.abs(ground_truth), axis=0)[-k:][
    :, np.newaxis
]
success_rate = np.zeros(len(M_phase_transition))
for index, m_phase_transition in enumerate(M_phase_transition):
    for _ in range(trials):
        Phi_pt = fm.Matrix(np.random.randn(m_phase_transition, N))
        alg_omp = fma.OMP(Phi_pt * Dict)
        b = Phi_pt * x
        recovered_support = np.argsort(
            np.abs(alg_omp.process(b, numMaxSteps=k)), axis=0
        )[-k:]
        if np.all(recovered_support == true_defect_positions):
            success_rate[index] += 1

success_rate = success_rate / trials

# Plot all results
plt.figure(1)
plt.clf()
plt.title("Ground Truth")
plt.plot(ground_truth)

plt.figure(2)
plt.clf()
plt.title("Actual Signal")
plt.plot(x_clean, label="Actual signal")
plt.plot(x, label="Actual signal with noise")
plt.legend()

plt.figure(3)
plt.clf()
plt.title("Observed Signal")
plt.plot(b)

plt.figure(4)
plt.clf()
plt.title("Reconstruction from M = " + str(M) + " measurements.")
plt.plot(ground_truth, label="Ground Truth")
plt.plot(y_omp, label="Reconstruction from OMP")
plt.plot(y_fista, label="Reconstruction from FISTA")
plt.plot(y_stela, label="Reconstruction from STELA")
plt.legend()
#
plt.figure(5)
plt.clf()
plt.title("Phase transition for sparsity k = " + str(k))
plt.plot(M_phase_transition / N, success_rate, label="Sucess rate of OMP")
plt.xlabel("compression ratio M/N")
plt.ylabel("Sucess rate")
plt.legend()
plt.show()
