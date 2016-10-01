import numpy as np


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def create_dataset(num_examples=10000, seq_length=64, period_range=(10,30)):

    min_count = int(np.ceil(seq_length/float(period_range[1])))
    max_count = int(np.ceil(seq_length/float(period_range[0])))
    count_range = max_count-min_count

    data   = np.zeros((num_examples, seq_length), dtype=np.float32)
    labels = np.zeros((num_examples, count_range), dtype=np.uint8)

    for i in range(num_examples):

        period = np.random.randint(period_range[0], period_range[1])
        cycle  = fftnoise(np.random.rand(period))
        signal = np.tile(cycle, int(np.ceil(seq_length/float(period))))
        signal = signal[0:seq_length]

        #mu = 0.0
        #sigma = np.std(signal) / 6.0
        #noise = sigma * np.random.randn(len(signal)) + mu

        data[i,]  = signal #+ noise

        # Set labels as one-hot encoding
        count = int(np.floor(seq_length / float(period)))
        labels[i,count-min_count] = 1 # <== note the -min_count

        #plt.plot(np.arange(seq_length), data[i,])
        #plt.show()
        #plt.clf()

    #scaler = MinMaxScaler(feature_range=(-1,+1))
    #data = scaler.fit_transform(data)
    data = np.expand_dims(data, axis=-1)

    return data, labels

num_train      = 50000
num_validation = 2000
num_test       = 2000
seq_length     = 64

trn_data, trn_labels = create_dataset(num_train, seq_length)
val_data, val_labels = create_dataset(num_validation, seq_length)
tst_data, tst_labels = create_dataset(num_test, seq_length)

np.save("./data/trn_inputs.npy", trn_data)
np.save("./data/trn_labels.npy", trn_labels)
np.save("./data/val_inputs.npy", val_data)
np.save("./data/val_labels.npy", val_labels)
np.save("./data/tst_inputs.npy", tst_data)
np.save("./data/tst_labels.npy", tst_labels)

print("Done generating data.")