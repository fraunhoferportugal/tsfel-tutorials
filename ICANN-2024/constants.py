import os
import pandas as pd

units = {"Accelerometer": "m/s2", "Electromyogram": "mV", "Goniometer": "degrees"}
# Sensor channel names
channels = ["Electromyogram 1", "Electromyogram 2", "Electromyogram 3", "Electromyogram 4",
            "Accelerometer Upper 1", "Accelerometer Upper 2", "Accelerometer Upper 3",
            "Goniometer 1",
            "Accelerometer Lower 1", "Accelerometer Lower 2", "Accelerometer Lower 3",
            "Goniometer 2"]

# Seven claseses for training and recogntion
classes = ["walk", "curve-left", "curve-right", "sit", "sit-to-stand", "stand", "stand-to-sit"]

dataset_folder = "Basic_CSL_2023"

# Unit conversion
def goniometer_unit_conversion(data, vcc=3, n=16):
    return ((vcc / (2 ** n - 1)) * data - (vcc / 2.0)) / (vcc / 2 * 606 * 10 ** -5)


def acc_unit_conversion(data, c_min=28000.0, c_max=38000.0):
    return ((data - c_min) / (c_max - c_min)) * 2 - 1


def emg_unit_conversion(data, vcc=3000, resolution=16, gain=1000):
    # Units are Volts
    return ((data / (2 ** resolution) - 0.5) * vcc) / gain


def unit_conversion(df):
    for column in df.columns:
        if 'Electromyogram' in column:
            df[column] = emg_unit_conversion(df[column])
        elif 'Accelerometer' in column:
            df[column] = acc_unit_conversion(df[column])
        elif 'Goniometer' in column:
            df[column] = goniometer_unit_conversion(df[column])
    return df


def get_train_test_data():

    # Set data fold and list all files
    train_folder = os.path.join(dataset_folder, "Training")
    test_folder = os.path.join(dataset_folder, "Testing")

    train_data, y_train = [], []
    for fl in os.listdir(train_folder):
        df = pd.read_csv(os.path.join(train_folder, fl), names=channels)     
        train_data += [df]# unit_conversion(df)]

        y_train += [fl.split('_')[0]]


    test_data, ids_test = [], []
    for fl in os.listdir(test_folder):
        df = pd.read_csv(os.path.join(test_folder, fl), names=channels)     
        test_data += [df] #unit_conversion(df)]

        ids_test += [fl.split('.')[0]]
    
    return train_data, y_train, test_data, ids_test