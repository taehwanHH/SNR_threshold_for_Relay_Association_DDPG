import numpy as np
import gym
import matlab.engine
from digital_comm_func import db2pow, pow2db, xi_dB, QAM_mod_Es, QAM_demod_Es
import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()

# given parameters
MOD = 6  # 2, 4, 6:  # of signal modulation bits: 1, 2, 4, 6, 8: (BPSK, QPSK, 16QAM, 64QAM, 256QAM):M=2^m# Sig Mod size
P_dBm = 32  # dBm
Kt = 40  # total number of RNs
P = 14  # 14, 24: number of pilots per CSI estimation
error_insertion = 1 # 0 = free, 1 = error
N = 1806 + P  # number of symbols per block
S = P  # max number of phase shift within a block[0, 1, ...]
P_bits = P  # BPSK for pilots
W = 25 * 10e3  # bandwidth 25 kHz
sigma2 = W * db2pow(-174) * 10 **(-3)  # noise power [Watt]
A = 1200  # area AxA m ^ 2
fc = 2  # carrier frequency[GHz]

# conv.enc. ---------------------------------------
trellis = eng.poly2trellis(matlab.double([5, 4]), matlab.double([[23, 35, 0], [0, 5, 13]]))  # R = 2 / 3
N_input = np.log2(trellis['numInputSymbols'])  # Number of input bit streams
N_output = np.log2(trellis['numOutputSymbols'])  # Number of output bit streams

coderate = float(N_input / N_output)
st2 = 4831  # States for random number
ConstraintLength = np.log2(trellis['numStates']) + 1
traceBack = np.ceil(7.5 * (ConstraintLength - 1))  # coding block size(bits)
Dsymb = N - P  # data symbol for a block
D_bits = int(MOD * Dsymb * coderate)
pilot_index = np.arange(0, P)
data_index = np.arange(P, N)
P_SN = db2pow(P_dBm) * 10 ** (-3)
P_RN = db2pow(P_dBm) * 10 ** (-3)
# ---------------------------------

# location coordinations
SNx = 0
SNy = 0
DNx = A
DNy = A

# location realization
RNx = A / 50 + (A - 2 * A / 50) * np.random.rand(Kt, 1)
RNy = A / 50 + (A - 2 * A / 50) * np.random.rand(Kt, 1)

# distances
dSR = np.sqrt(RNx ** 2 + RNy ** 2)
dRD = np.sqrt((RNx - DNx) ** 2 + (RNy - DNy) ** 2)


def calculate_err(eta):
    # one block of frame
    tx_bits = np.random.randint(2, size=(D_bits, 1))  # data bits for one block
    tx_bits_enc = eng.convenc(tx_bits, trellis)
    tx_bits_enc_inter = eng.randintrlv(tx_bits_enc, st2)

    p_bits = np.ones(shape=(1,P_bits))  # pilot bits for the first hop
    x_d = QAM_mod_Es(tx_bits_enc_inter, MOD)
    x = np.append(p_bits, x_d)  # BPSK for pilots

    # one block of channel
    g = np.sqrt(db2pow(xi_dB(dSR))) * np.sqrt(1 / 2) * (np.random.randn(Kt, 1) + 1j * np.random.randn(Kt, 1))
    h = np.sqrt(db2pow(xi_dB(dRD))) * np.sqrt(1 / 2) * (np.random.randn(Kt, 1) + 1j * np.random.randn(Kt, 1))

    # RN association
    # calculate received SNRs at RNs
    K_ind = np.argwhere(pow2db(abs(np.sqrt(P_SN) * g) ** 2 / sigma2) >= eta)  # index of active RNs
    K_ind = K_ind[:, 0]
    K = len(K_ind)

    if K < 1:
        err_w_PRb = D_bits * coderate  # if no RN is associated, set all errors

    else:
        ga = g[K_ind]  # 1st - link channel of active RNs
        ha = h[K_ind]  # 2nd - link channel of active RNs
        ## The 1st Phase
        zr = error_insertion * np.sqrt(sigma2 / 2) * (np.random.randn(K, N) + 1j * np.random.randn(K, N))  # noise generation
        yr = ga * np.sqrt(P_SN) * x + zr  # Rx signals at the active RNs

        ## w / o PR: conventional method(benchmarking)
        # channel estimation at RNs via P pilots
        if error_insertion == 0:
            ga_hat_wo_PR = ga * np.sqrt(P_SN)
        else:
            ga_hat_wo_PR = np.mean(yr[:, 0: P], 1)  # via P pilots

        # equalization
        x_d_hat_RN_K_wo_PR = yr[:, data_index] / np.kron(np.ones(shape=(1, Dsymb)), ga_hat_wo_PR.reshape((-1,1)))
        x_dr_wo_PR = []

        # regeneration per DF - RN
        for k in range(0, K):
            x_d_hat_RN_wo_PR = x_d_hat_RN_K_wo_PR[k, :]
            D_bit_hat_RN_wo_PR = QAM_demod_Es(x_d_hat_RN_wo_PR, MOD)
            x_dr_wo_PR.append(QAM_mod_Es(D_bit_hat_RN_wo_PR, MOD).tolist())

        # AWGN at DN
        zd = error_insertion * np.sqrt(sigma2 / 2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # retransmission
        h_wo_PR = np.kron(np.ones(shape=(1, N)), ha)

        ## w / PR: Proposed method
        # channel estimation at RNs via PxS pilots the same as w / o PR
        # pilot insertion & regeneration
        x_stackr_w_PR = np.zeros(shape=(K, N),dtype=complex)
        x_stackr_w_PR[:, data_index] = np.array(x_dr_wo_PR)
        theta_tmp = 2 * np.pi * np.random.rand(K, S)
        x_stackr_w_PR[:, pilot_index] = 1

        theta_data = np.kron(theta_tmp, np.ones(shape=(1, int((N - P) / S))))
        theta_pilot = np.kron(theta_tmp, np.ones(shape=(1, int(P / S))))
        theta = np.append(theta_pilot, theta_data, axis=1)
        h_w_PR = h_wo_PR * np.exp(1j * theta)  # PR: effective ch
        ydo = np.sum(h_w_PR * np.sqrt(P_RN) * x_stackr_w_PR, 0).reshape(1,-1) + zd  # Rx signals at the active RNs

        # channel estimation at DN
        if error_insertion == 0:
            ha_w_PR_hat = np.sum(np.kron(ha, np.ones(shape=(1, S))) * np.exp(1j * theta_tmp), 0) * np.sqrt(P_RN)
        else:
            ha_w_PR_hat = np.mean(np.reshape(ydo[0, pilot_index], (int(P / S), S)), 0)

        # channel equalization
        x_d_hat_DN_stacko = ydo[:, data_index] / np.kron(ha_w_PR_hat, np.ones(shape=(1, int((N - P) / S))))
        # demodulation
        x_d_hat_DNo = x_d_hat_DN_stacko
        D_bit_hat_DNo = QAM_demod_Es(x_d_hat_DNo, MOD)
        D_bit_hat_DNo_deinter = eng.randdeintrlv(D_bit_hat_DNo, st2)  # Deinterleave
        D_bit_hat_DNo_decoded = eng.vitdec(D_bit_hat_DNo_deinter, trellis, traceBack, 'trunc', 'hard')


        # error check at DN
        err_w_PRb = sum(abs(np.array(D_bit_hat_DNo_decoded).reshape(-1) - tx_bits.reshape(-1)))

    return err_w_PRb

def mean_err(eta, iter_num=10):
    iter = np.arrange(iter_num)
    s = 0
    for i in iter:
        s += calculate_err(eta)
    return s/iter_num


class CommunicationEnv:
    def __init__(self, eta_upper, target_ber, noise_var):
        self.target_ber = target_ber
        self.noise_var = noise_var

        # Define the observation space and action space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=eta_upper, shape=(1,))


    def step(self, action):
        # Convert the action from a tensor to a numpy array
        action = action.detach().numpy()

        # Add noise to the action
        action += np.random.normal(0, 0.1, size=action.shape)

        # Clip the action to the action space limits
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Calculate the bit error rate (BER) based on the action value
        ber = mean_err(action)

        # Calculate the reward based on the BER and target BER
        reward = -np.abs(ber - self.target_ber)

        # Generate the next observation based on the current observation and action
        obs = np.random.rand(self.n_state)

        # Add noise to the observation
        obs += np.random.normal(0, self.noise_var, size=obs.shape)

        # Clip the observation to the observation space limits
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        # Check if the episode is done (i.e., if the BER is below the target BER)
        done = ber <= self.target_ber

        return obs, reward, done, {}

    def reset(self):
        # Generate a random observation to start the episode
        obs = np.random.rand(self.n_state)

        # Add noise to the observation
        obs += np.random.normal(0, self.noise_var, size=obs.shape)

        # Clip the observation to the observation space limits
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs












