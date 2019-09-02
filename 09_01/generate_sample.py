import numpy as np
from typing import Optional, Tuple


def generate_sample(f: Optional[float] = 1.0, t0: Optional[float] = None, batch_size: int = 1,
                    predict: int = 50, samples: int = 100, ninputs: int = 1, noutputs: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates data samples.

    :param f: The frequency to use for all time series or None to randomize.
    :param t0: The time offset to use for all time series or None to randomize.
    :param batch_size: The number of time series to generate.
    :param predict: The number of future samples to generate.
    :param samples: The number of past (and current) samples to generate.
    :return: Tuple that contains the past times and values as well as the future times and values. In all outputs,
             each row represents one time series of the batch.
    """
    Fs = 100

    T = np.empty((batch_size, samples))
    Y = np.empty((batch_size, samples, ninputs))
    Y_output = np.empty((batch_size, samples, noutputs))
    FT = np.empty((batch_size, predict))
    FY = np.empty((batch_size, predict, noutputs))

    _t0 = t0
    for i in range(batch_size):
        t = np.arange(0, samples + predict) / Fs
        if _t0 is None:
            t0 = np.random.rand() * 2 * np.pi
        else:
            t0 = _t0 + i/float(batch_size)

        freq = f
        if freq is None:
            freq = np.random.rand() * 3.5 + 0.5

        y = np.sin(2 * np.pi * freq * (t + t0))
        y_cos = np.cos(2 * np.pi * freq * (t + t0))
        y_out1 = np.sin(2 * np.pi * freq * (t + t0))
        y_out2 = np.cos(4 * np.pi * freq * (t + t0))
        y_out3 = np.cos(8 * np.pi * freq * (t + t0))
        y_out4 = np.sin(8 * np.pi * freq * (t + t0))

        y_transp = np.transpose(np.array([y, y_cos]))
        y_transp_output = np.transpose(np.array([y_out1, y_out2, y_out3, y_out4]))

        T[i, :] = t[0:samples]
        Y[i, :, :] = y_transp[0:samples, :]
        Y_output[i, :, :] = y_transp_output[0:samples, :]

        FT[i, :] = t[samples:samples + predict]
        FY[i, :] = y_transp_output[samples:samples+predict, :]

    return T, Y, FT, FY, Y_output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    import seaborn as sns

    t, y, t_next, y_next = generate_sample(f=None, t0=None, batch_size=3)

    n_tests = t.shape[0]
    for i in range(0, n_tests):
        plt.subplot(n_tests, 1, i+1)
        plt.plot(t[i, :], y[i, :])
        plt.plot(np.append(t[i, -1], t_next[i, :]), np.append(y[i, -1], y_next[i, :]), color='red', linestyle=':')

    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.show()