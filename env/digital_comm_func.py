import numpy as np


def db2pow(db):
    return 10 ** (db / 10)


def pow2db(pow):
    return 10*np.log10(pow)


def xi_dB(dist, fc=2):
    return - 12.7 - 26 * np.log10(fc) - 36.7 * np.log10(dist)


def QAM_mod_Es(data, bit):
    data = np.array(data)
    N_input = data.size
    d = np.reshape(np.array(data), (bit, int(N_input / bit)))

    if bit == 1:  # BPSK
        a = 1
        x_real = 2 * d[0, :] - 1
        x_imag = np.zeros(size=(1, N_input))

    elif bit == 2:  # QPSK"1" --> +, "0" --> -,
        a = 1 / np.sqrt(2)
        x_real = (2 * d[0, :] - 1)
        x_imag = (2 * d[1, :] - 1)

    elif bit == 4:  # 16QAM
        a = 1 / np.sqrt(10)
        x_real = np.multiply((2 * d[0, :] - 1), (2 * d[1, :] - 1 + 2))  # (sign) * (1, 3)
        x_imag = np.multiply((2 * d[2, :] - 1), (2 * d[3, :] - 1 + 2))

    elif bit == 6:  # 64QAM
        a = 1 / np.sqrt(42)
        x_real = (2 * d[0, :] - 1) * ((2 * d[1, :] - 1) * (2 * d[2, :] + 1) + 4)  # (sign) * ((sign) * (1, 3) + 4) = ( sign) * (1,3,5,7)
        x_imag = (2 * d[3, :] - 1) * ((2 * d[4, :] - 1) * (2 * d[5, :] + 1) + 4)

    return (x_real + 1j * x_imag) * a


def QAM_demod_Es(y, bit):
    # QAM_demod  QAM demodulation
    #   QAM_demod(y,bit) detect binary input bits from received signal "y"
    #   "bit" is the number of bits for each modulated symbol and the value
    #   of "bit" is 1, 2, 4 or 6.
    # Input bit is "0" or "1" row vector
    # Normalize the symbol energy to "1"
    y = np.array(y)
    N_input = y.size
    d_hat = np.zeros(shape=(bit, N_input))

    if  bit == 1:
        d_hat[0,:]=(np.sign(np.real(y))+1)/2

    elif bit == 2:
        a = 1/np.sqrt(2)
        d_hat[0,:]=(np.sign(np.real(y))+1)/2 # "+" --> 1 , # "-" --> 0
        d_hat[1,:]=(np.sign(np.imag(y))+1)/2

    elif bit == 4:
        a=1/np.sqrt(10)
        d_hat[0,:]=(np.sign(np.real(y))+1)/2 # + --> 1, - --> 0
        d_hat[1,:]=(np.sign(abs(np.real(y))-2*a)+1)/2
        d_hat[2,:]=(np.sign(np.imag(y))+1)/2
        d_hat[3,:]=(np.sign(abs(np.imag(y))-2*a)+1)/2

    elif bit == 6:
        a=1/np.sqrt(42)
        d_hat[0,:]=(np.sign(np.real(y))+1)/2
        d_hat[1,:]=(np.sign(abs(np.real(y))-4*a)+1)/2
        d_hat[2,:]=(np.sign(abs(abs(np.real(y))-4*a)-2*a)+1)/2
        d_hat[3,:]=(np.sign(np.imag(y))+1)/2
        d_hat[4,:]=(np.sign(abs(np.imag(y))-4*a)+1)/2
        d_hat[5,:]=(np.sign(abs(abs(np.imag(y))-4*a)-2*a)+1)/2

    return np.reshape(d_hat, (1, bit*N_input))
