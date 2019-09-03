import numpy as np
from sklearn import preprocessing 
from scipy import signal

def data_read():
    # reading pointcloud data
    content = open("../data/image_processing_dots_lines.txt", "r").read().splitlines()
    content_numerical = []
    for t in range(len(content)):
        timestep = []
        for i in range(len(content[t].split('],'))):
            try:
                line = [int(content[t].split('],')[i].split(',')[0].split('[[')[1]), int(content[t].split('],')[i].split(',')[1])]
            except:
                try:
                    line = [int(content[t].split('],')[i].split(',')[0].split('[')[1]), int(content[t].split('],')[i].split(',')[1])]
                except:
                    line = [int(content[t].split('],')[i].split(',')[0].split('[')[1]), int(content[t].split('],')[i].split(',')[1].split(']]')[0])]
            timestep.append(line)
        content_numerical.append(timestep)

    timeseries_Y = [[0 for t in range(len(content_numerical))] for i in range(100)]

    for t in range(len(content_numerical)):
        for i in range(100):
            timeseries_Y[i][t] = content_numerical[t][i]

    timeseries_Y_scaled = []

    for items in timeseries_Y:
        timeseries_Y_scaled.append(preprocessing.scale(items))
        # timeseries_Y_scaled.append(items)

    timeseries_Y_scaled = np.array(timeseries_Y_scaled)

    timeseries_Y_scaled_flatten = [[0 for t in range(4200)] for i in range(200)]

    for ii in range(100):
        for jj in range(4200):
            timeseries_Y_scaled_flatten[ii][jj] = timeseries_Y_scaled[ii][jj][0]
            timeseries_Y_scaled_flatten[ii+100][jj] = timeseries_Y_scaled[ii][jj][1]

    timeseries_Y_scaled_flatten = np.array(timeseries_Y_scaled_flatten)

    # reading Skinflow data
    content = open("../data/image_processing_skinflow_lines.txt", "r").read().splitlines()
    content_numerical = []

    for t in range(len(content)):
        line = []
        for i in range(16):
            if i == 0:
                line.append(int(content[t].split(',')[0].split('[')[1]))
            elif i == 15 :
                line.append(int(content[t].split(',')[15].split(']')[0]))
            else:
                line.append(int(content[t].split(',')[i]))
        content_numerical.append(line)

    timeseries_X= [[0 for i in range(len(content_numerical))] for j in range(len(content_numerical[0]))]

    for i in range(len(timeseries_X)):
        for j in range(len(timeseries_X[0])):
            timeseries_X[i][j] = content_numerical[j][i]

    timeseries_X_scaled = []

    for items in timeseries_X:
        timeseries_X_scaled.append(preprocessing.scale(items))

    return timeseries_X_scaled, timeseries_Y_scaled_flatten

def generate_sample(test_time_0, X_read, Y_read,f = 1.0, t0 = None, batch_size = 1, predict = 50, samples = 100, ninputs = 16, noutputs = 200):

    X_read = np.array(X_read)
    Y_read = np.array(Y_read)

    X_batch = np.empty((batch_size, samples, ninputs))
    Y_batch = np.empty((batch_size, predict, noutputs))

    X_test = np.empty((batch_size, samples, ninputs))
    Y_test = np.empty((batch_size, predict, noutputs))

    Y_before_t = np.empty((batch_size, samples, noutputs))

    _t0 = t0
    for i in range(batch_size):
        if _t0 is None:
            t0 = np.random.randint(4000-samples-1)
            t0_test = test_time_0
        else:
            t0 = _t0 + i/float(batch_size)

        time = np.linspace(0, 4199, num = 4200)
        X_test_time = np.linspace(3999+t0_test, 3999+t0_test+samples, num = samples)
        Y_test_time = np.linspace(3999+t0_test+samples, 3999+t0_test+samples+predict, num = predict)
        X_time = np.linspace(t0, t0+samples, num = samples)
        Y_time = np.linspace(t0+samples, t0+samples+predict, num = predict)

        X_transp = np.transpose(np.array(X_read))
        Y_transp = np.transpose(np.array(Y_read))

        X_batch[i, :, :] = X_transp[t0:t0+samples, :]
        Y_batch[i, :] = Y_transp[t0+samples:t0+samples+predict, :]

        X_test[i, :, :] = X_transp[4000+t0_test:4000+t0_test+samples, :]
        Y_test[i, :] = Y_transp[4000+t0_test+samples:4000+t0_test+samples+predict, :]

        Y_before_t[i, :, :] = Y_transp[t0:t0+samples, :]

    return X_time, Y_time, X_batch, Y_batch, Y_before_t, time, X_transp, Y_transp, X_test_time, Y_test_time, X_test, Y_test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    import seaborn as sns

    Xread, Yread = data_read()
    x_t, y_t, x_batch, y_batch, y_before, time, x_all, y_all, x_test_time, y_test_time, x_test, y_test = generate_sample(0, X_read = Xread, Y_read = Yread, f = 1, t0 = None, samples = 10, predict = 1, ninputs = 16, noutputs = 200)

    print('xtshape: ', x_t.shape,'ytshape: ', y_t.shape,'xbatchshape: ', x_batch.shape,'ybatchshape: ', y_batch.shape)

    plt.subplot(211)
    plt.plot(time,x_all, color = 'gray')
    plt.plot(x_t,x_batch.squeeze(), color = 'green')
    plt.plot(x_test_time,x_test.squeeze(), color = 'blue')
    plt.subplot(212)
    plt.plot(time,y_all, color = 'gray')
    plt.plot(y_t,y_batch.squeeze(axis = 0), color = 'green', marker ='o')
    plt.plot(y_test_time,y_test.squeeze(axis = 0), color = 'blue', marker = 'o')
    plt.show()

    # t, y, t_next, y_next = generate_sample(f=None, t0=None, batch_size=3)

    # n_tests = t.shape[0]
    # for i in range(0, n_tests):
    #     plt.subplot(n_tests, 1, i+1)
    #     plt.plot(t[i, :], y[i, :])
    #     plt.plot(np.append(t[i, -1], t_next[i, :]), np.append(y[i, -1], y_next[i, :]), color='red', linestyle=':')

    # plt.xlabel('time [t]')
    # plt.ylabel('signal')
    # plt.show()