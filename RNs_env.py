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

def db2pow(db):
    return 10 ** (db / 10)
