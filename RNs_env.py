import numpy as np
import gym


class CommunicationEnv:
    def __init__(self, eta_upper, target_ber, noise_var):

        self.target_ber = target_ber
        self.noise_var = noise_var

        # Define the observation space and action space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=eta_upper, shape=(1,))
     
        MOD = 6 # 2, 4, 6:  # of signal modulation bits: 1, 2, 4, 6, 8: (BPSK, QPSK, 16QAM, 64QAM, 256QAM):M=2^m# Sig Mod size
        P_dBm = 32 # dBm
        Kt = 40 # total number of RNs
        P = 14 # 14, 24: number of pilots per CSI estimation
        error_insertion = 1 # 0 = free, 1 = error
        N = 1806 + P # number of symbols per block
        S = P # max number of phase shift within a block[0, 1, ...]
        P_bits = P # BPSK for pilots
        W = 25 * 10e3 # bandwidth 25 kHz
        sigma2 = W * db2pow(-174) * 10 ^ (-3) # noise power [Watt]
        A = 1200 # area AxA m ^ 2
        fc = 2 # carrier frequency[GHz]
        self.fc = fc

        # conv.enc. ---------------------------------------
        trellis = poly2trellis([5 4], [23 35 0 0 5 13]) # R = 2 / 3
        N_input = np.log2(trellis.numInputSymbols) # Number of input bit streams
        N_output = np.log2(trellis.numOutputSymbols) # Number of output bit streams
        coderate = N_input / N_output
        st2 = 4831 # States for random number
        ConstraintLength = np.log2(trellis.numStates) + 1
        traceBack = np.ceil(7.5 * (ConstraintLength - 1)) # coding block size(bits)
        Dsymb = N - P # data symbol for a block
        D_bits = MOD * Dsymb * coderate
        pilot_index = np.arrange(1, P+1)
        data_index = np.arrange(P+1, N+1)
        P_SN = db2pow(P_dBm) * 10 ^ (-3)
        P_RN = db2pow(P_dBm) * 10 ^ (-3)
        # ---------------------------------

    def step(self, action):
        # Convert the action from a tensor to a numpy array
        action = action.detach().numpy()

        # Add noise to the action
        action += np.random.normal(0, 0.1, size=action.shape)

        # Clip the action to the action space limits
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Calculate the bit error rate (BER) based on the action value
        ber = calculate_ber(action)

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


    def xi_dB(self,dist):
        return - 12.7 - 26 * np.log10(self.fc) - 36.7 * np.log10(dist)


    def calculate_ber(self, eta):
# one block of frame
tx_bits = randi([0 1], D_bits, 1) # data bits for one block
tx_bits_enc = convenc(tx_bits, trellis)
tx_bits_enc_inter = randintrlv(tx_bits_enc, st2)
p_bits = ones(1, P_bits) # pilot bits for the first hop
x_d = QAM_mod_Es(tx_bits_enc_inter, MOD)
x = [p_bits x_d] # BPSK for pilots

# one block of channel
g = sqrt(db2pow(xi_dB(dSR))). * sqrt(1 / 2). * (randn(Kt, 1) + 1i * randn(Kt, 1))
h = sqrt(db2pow(xi_dB(dRD))). * sqrt(1 / 2). * (randn(Kt, 1) + 1i * randn(Kt, 1))

# RN association
# calculate received SNRs at RNs
K_ind = find(pow2db(abs(sqrt(P_SN) * g). ^ 2. / sigma2) >= eta) # index of active RNs
K = length(K_ind)
if K < 1:
err_w_PRb(b) = D_bits * coderate # if no RN is associated, set all errors
else :

ga = g(K_ind) # 1st - link channel of active RNs
ha = h(K_ind) # 2nd - link channel of active RNs

## The 1st Phase
zr = error_insertion * sqrt(sigma2 / 2) * (randn(K, N) + 1i * randn(K, N)) # noise generation
yr = ga * sqrt(P_SN) * x + zr # Rx signals at the active RNs
## w / o PR: conventional method(benchmarking)
# channel estimation at RNs via P pilots
if error_insertion == 0:
    ga_hat_wo_PR = ga * sqrt(P_SN)
else:
    ga_hat_wo_PR = mean(yr(:, 1: P), 2) # via P pilots

# equalization
x_d_hat_RN_K_wo_PR = yr(:, data_index)./ kron(ones(1, Dsymb), ga_hat_wo_PR)
x_dr_wo_PR = []
# regeneration per DF - RN
for k=1:K
x_d_hat_RN_wo_PR = x_d_hat_RN_K_wo_PR(k,:)
D_bit_hat_RN_wo_PR = QAM_demod_Es(x_d_hat_RN_wo_PR, MOD)
x_dr_wo_PR = [x_dr_wo_PR; QAM_mod_Es(D_bit_hat_RN_wo_PR, MOD)]


# AWGN at DN
zd = error_insertion * sqrt(sigma2 / 2) * (randn(1, N) + 1i * randn(1, N))
# retransmission
h_wo_PR = kron(ones(1, N), ha)

## w / PR: Proposed method
# channel estimation at RNs via PxS pilots thesame as w / o PR
# pilot insertion & regeneration
x_stackr_w_PR = zeros(K, N)
x_stackr_w_PR(:, data_index) = x_dr_wo_PR
theta_tmp = 2 * pi * rand(K, S)
x_stackr_w_PR(:, pilot_index) = 1

theta_data = kron(theta_tmp, ones(1, (N - P) / S))
theta_pilot = kron(theta_tmp, ones(1, P / S))
theta = [theta_pilot theta_data]
h_w_PR = h_wo_PR. * exp(1i * theta) # PR: effective ch

ydo = sum(h_w_PR. * sqrt(P_RN). * x_stackr_w_PR, 1) + zd # Rx signals at the active RNs

# channel estimation at DN
if error_insertion == 0:
    ha_w_PR_hat = sum(kron(ha, ones(1, S)). * exp(1i * theta_tmp), 1)*sqrt(P_RN)
else:
    ha_w_PR_hat = mean(reshape(ydo(1, pilot_index), P / S, S), 1)

# channel equalization
x_d_hat_DN_stacko = ydo(:, data_index)./ kron(ha_w_PR_hat, ones(1, (N - P) / S))
# demodulation
x_d_hat_DNo = x_d_hat_DN_stacko
D_bit_hat_DNo = QAM_demod_Es(x_d_hat_DNo, MOD)
D_bit_hat_DNo_deinter = randdeintrlv(D_bit_hat_DNo, st2) # Deinterleave
D_bit_hat_DNo_decoded = vitdec(D_bit_hat_DNo_deinter, trellis, traceBack, 'trunc', 'hard')

# error check at DN
err_w_PRb(b) = sum(abs(D_bit_hat_DNo_decoded - tx_bits'))


def db2pow(db):
    return 10 ** (db / 10)
