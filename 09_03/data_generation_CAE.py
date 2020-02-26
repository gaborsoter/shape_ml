import numpy as np
from sklearn import preprocessing 
from scipy import signal
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

mpl.rc('legend', fontsize=12)

mpl.rcParams['xtick.labelsize'] =  7
mpl.rcParams['ytick.labelsize'] =  7
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 7
plt.rcParams["figure.figsize"] = [3.5,2.5]
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'

fig = plt.figure(figsize=(3.5, 2.5)) 

def baseline_shifting(input_array, where = []):
    output_array = np.array([])
    first_array  = input_array[0:where[0]]
    s2 = np.std(first_array)
    m2 = np.mean(first_array)
    output_array = np.append(output_array, first_array)
    for i in range(len(where)):
        if len(where) == 1:
            i_array = input_array[where[0]:]
        if i == len(where)-1:
            i_array = input_array[where[i]:]
        else:
            i_array = input_array[where[i]:where[i+1]]

        s1 = np.std(i_array)
        m1 = np.mean(i_array)    
        for j in range(len(i_array)):
            i_array[j] = m2 + (i_array[j] - m1) * s2 / s1

        output_array = np.append(output_array, np.array(i_array))

    return output_array

def scaling_single_feature(array):
    numpyify = np.array(array)
    array_reshaped = numpyify.reshape(-1,1)
    scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    scaler.fit(array_reshaped)
    scaled = scaler.transform(array_reshaped).reshape(numpyify.shape[0])
    return scaled

def data_read():
    # reading pointcloud data
    content = open("../data/data_CAE_encoded.txt", "r").read().splitlines()
    content_numerical = []
    for t in range(len(content)):
        line = content[t].split(' ')
        timestep = []
        for i in range(len(line)-1):
            timestep.append(float(line[i]))
        content_numerical.append(timestep) 

    timeseries_Y = [[0 for t in range(len(content_numerical))] for i in range(324)]

    for t in range(len(content_numerical)):
        for i in range(324):
            timeseries_Y[i][t] = content_numerical[t][i]

    timeseries_Y_scaled = []

    array_zero = []
    nonzero = 0
    for i in range(len(timeseries_Y)):
        if timeseries_Y[i][0] != 0.0:
            nonzero += 1
            # timeseries_Y_scaled.append(preprocessing.scale(scaling_single_feature(items)))
            timeseries_Y_scaled.append(timeseries_Y[i])
        else:
            array_zero.append(i)

    # filtering
    fs = 20
    fc = 2  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(1, w, 'low')

    for i in range(len(timeseries_Y_scaled)):
        # timeseries_Y_scaled[i] = signal.filtfilt(b, a,timeseries_Y_scaled[i])
        timeseries_Y_scaled[i] = timeseries_Y_scaled[i]

    print('nonzero: ', nonzero)
    print('zero array: ', array_zero)

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
        timeseries_X_scaled.append(scaling_single_feature(items))

    timeseries_X_scaled = np.array(timeseries_X_scaled)
    timeseries_X_before_filtering = np.copy(timeseries_X_scaled)

    # drift compensation and filtering

    # for i in range(timeseries_Y_scaled_flatten.shape[0]):
    #     timeseries_Y_scaled_flatten[i] = signal.filtfilt(b, a,baseline_shifting(timeseries_Y_scaled_flatten[i], where = [2053,2850]))

    base0 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[0,:], where = [503, 1042, 1517, 2159, 3000]))
    base1 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[1,:], where = [1535, 2770]))
    base2 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[2,:], where = [3195]))
    base3 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[3,:], where = [2765]))
    base4 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[4,:], where = [1900, 2800]))
    base5 = signal.filtfilt(b, a,timeseries_X_scaled[5,:])
    base6 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[6,:], where = [330, 2214]))
    base7 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[7,:], where = [1290, 2059]))
    base8 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[8,:], where = [1563, 2794]))
    base9 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[9,:], where = [919, 1169, 2476]))
    base10 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[10,:], where = [386, 927, 1965, 3333]))
    base11 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[11,:], where = [239, 1141, 1625, 2672, 3430]))
    base12 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[12,:], where = [865, 2163, 3440]))
    base13 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[13,:], where = [619, 1570, 2600, 3399]))
    base14 = signal.filtfilt(b, a,timeseries_X_scaled[14,:])
    base15 = signal.filtfilt(b, a,baseline_shifting(timeseries_X_scaled[15,:], where = [448, 1332, 2483]))
    
    for i in range(len(timeseries_X_scaled[0])):
        timeseries_X_scaled[0,i] = scaling_single_feature(base0)[i]
        timeseries_X_scaled[1,i] = scaling_single_feature(base1)[i]
        timeseries_X_scaled[2,i] = scaling_single_feature(base2)[i]
        timeseries_X_scaled[3,i] = scaling_single_feature(base3)[i]
        timeseries_X_scaled[4,i] = scaling_single_feature(base4)[i]
        timeseries_X_scaled[5,i] = scaling_single_feature(base5)[i]
        timeseries_X_scaled[6,i] = scaling_single_feature(base6)[i]
        timeseries_X_scaled[7,i] = scaling_single_feature(base7)[i]
        timeseries_X_scaled[8,i] = scaling_single_feature(base8)[i]
        timeseries_X_scaled[9,i] = scaling_single_feature(base9)[i]
        timeseries_X_scaled[10,i] = scaling_single_feature(base10)[i]
        timeseries_X_scaled[11,i] = scaling_single_feature(base11)[i]
        timeseries_X_scaled[12,i] = scaling_single_feature(base12)[i]
        timeseries_X_scaled[13,i] = scaling_single_feature(base13)[i]
        timeseries_X_scaled[14,i] = scaling_single_feature(base14)[i]
        timeseries_X_scaled[15,i] = scaling_single_feature(base15)[i]

    return timeseries_X_scaled, timeseries_Y_scaled, array_zero

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
        X_test_time = np.linspace(3799+t0_test, 3799+t0_test+samples, num = samples)
        Y_test_time = np.linspace(3799+t0_test+samples, 3799+t0_test+samples+predict, num = predict)
        X_time = np.linspace(t0, t0+samples, num = samples)
        Y_time = np.linspace(t0+samples, t0+samples+predict, num = predict)

        X_transp = np.transpose(np.array(X_read))
        Y_transp = np.transpose(np.array(Y_read))

        X_batch[i, :, :] = X_transp[t0:t0+samples, :]
        Y_batch[i, :] = Y_transp[t0+samples:t0+samples+predict, :]

        X_test[i, :, :] = X_transp[3999+t0_test:3999+t0_test+samples, :]
        Y_test[i, :] = Y_transp[3999+t0_test+samples:3999+t0_test+samples+predict, :]

        Y_before_t[i, :, :] = Y_transp[t0:t0+samples, :]

    return X_time, Y_time, X_batch, Y_batch, Y_before_t, time, X_transp, Y_transp, X_test_time, Y_test_time, X_test, Y_test

if __name__ == '__main__':

    Xread, Yread, zerooo = data_read()

    Xread = np.array(Xread)
    Yread = np.array(Yread)

    # this is really cool for feature matching
    plt.plot(Yread[10,:])
    plt.plot(Xread[10,:])
    plt.show()

    # print('xshape: ', Xread.shape)
    # print('yshape: ', Yread.shape)

    # # print(np.array(Xread)[2,:])

    # fs = 20
    # fc = .5  # Cut-off frequency of the filter
    # w = fc / (fs / 2) # Normalize the frequency

    # # for points in range(len(Xread)):# finish this (filtering...)
    # normal = np.copy(Xread[15,:])
    # b, a = signal.butter(1, w, 'low')
    # filtered = signal.filtfilt(b, a, normal)

    # # print(baseline_shifting(Yread[30,:], where = [2053,2850]))   

    # plt.plot(normal, color = 'gray')
    # # plt.plot(filtered, color = 'blue')
    # # plt.plot(baseline_shifting(Xread[1,:], where = [503, 1042, 1517, 2159, 3000]), color = 'blue')


    # plt.show()

    # x_t, y_t, x_batch, y_batch, y_before, time, x_all, y_all, x_test_time, y_test_time, x_test, y_test = generate_sample(0, X_read = Xread, Y_read = Yread, f = 1, t0 = None, samples = 10, predict = 1, ninputs = 16, noutputs = 200)

    # print('xtshape: ', x_t.shape,'ytshape: ', y_t.shape,'xbatchshape: ', x_batch.shape,'ybatchshape: ', y_batch.shape)

    # plt.subplot(211)
    # ax = fig.add_subplot(211)
    # ax.set_xlim([500, 1000])
    # ax.set_ylim([-3, 3])
    # ax.plot(time,x_all[:,1])
    # ax.set_xlabel('Time', fontsize=7)
    # ax.set_ylabel('Normalised liquid \n displacement', fontsize=7)
    # ax.tick_params(direction='out', length=2, width=1, which='both', axis='both')

    # ax2 = fig.add_subplot(212)
    # ax2.set_xlim([500, 1000])
    # ax2.set_ylim([-3, 3])
    # ax2.plot(time,y_all[:,1])
    # ax2.set_xlabel('Time', fontsize=7)
    # ax2.set_ylabel('Normalised horizontal \n position', fontsize=7)
    # ax2.tick_params(direction='out', length=2, width=1, which='both', axis='both')
    # plt.tight_layout()
    # plt.show()
    # fig.savefig("timeseries.pdf")
