import os
import glob
import h5py
import numpy as np
import pandas as pd

CHANNELS = [
    "EMG1",
    "EMG2",
    "EMG3",
    "EMG4",
    "Airborne",
    "ACC upper X",
    "ACC upper Y",
    "ACC upper Z",
    "Goniometer X",
    "ACC lower X",
    "ACC lower Y",
    "ACC lower Z",
    "Goniometer Y",
    "GYRO upper X",
    "GYRO upper Y",
    "GYRO upper Z",
    "GYRO lower X",
    "GYRO lower Y",
    "GYRO lower Z",
]


raw_dir = '../datasets/csl_share_liu_et_al'
proc_dir = 'New_users/'

fls = sorted(glob.glob(raw_dir + os.sep + "*" + os.sep))

samples = []
df_data = pd.DataFrame()
for fl in fls:
    acquisition_files = sorted(glob.glob(os.path.join(fl, "*.h5")))

    user = fl.split(os.sep)[-2]
    print("USER: ", user)

    for acq_fl in acquisition_files:

        # load annotation
        annotation = pd.read_csv(acq_fl[: -len(".h5")] + ".csv", header=None)

        # load sensors data
        with h5py.File(acq_fl, "r") as f:
            a_group_key = list(f.keys())[0]
            d = np.array(f[a_group_key])
        raw_data = pd.DataFrame(d, columns=CHANNELS)

        # remove first point
        raw_data = raw_data.iloc[1:]
        raw_data = raw_data.drop(columns=CHANNELS[4:5] + CHANNELS[13:])

        for i, idx in annotation.iterrows():
            repetiton = raw_data.loc[idx[1] : idx[2]]
            filename = proc_dir + idx[0] + '_' +  user + '-' + str(i) + ".csv"
            repetiton.to_csv(filename, index=False, header=False)
