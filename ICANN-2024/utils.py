import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Dictionary containing the units of measurement for each sensor type
units = {
    "Accelerometer": "g",
    "Electromyogram": "mV",
    "Goniometer": "degrees"
}

# List of sensor channels collected from the dataset
channels = [
    "Electromyogram 1", "Electromyogram 2", "Electromyogram 3", "Electromyogram 4",
    "Accelerometer Upper 1", "Accelerometer Upper 2", "Accelerometer Upper 3",
    "Goniometer 1",
    "Accelerometer Lower 1", "Accelerometer Lower 2", "Accelerometer Lower 3",
    "Goniometer 2"
]

# Activity classes used for classification
classes = [
    "walk", "walk-curve-left", "walk-curve-right", "sit",
    "sit-to-stand", "stand", "stand-to-sit"
]

# Root directory for the dataset
dataset_folder = "Basic_CSL"


def goniometer_unit_conversion(data, vcc=3, n=16):
    """
    Convert raw goniometer data to angular degrees.
    
    :param data: Raw goniometer sensor data
    :param vcc: Reference voltage for the sensor (default: 3V)
    :param n: Bit resolution of the sensor (default: 16 bits)
    :return: Converted data in degrees
    """
    return ((vcc / (2 ** n - 1)) * data - (vcc / 2.0)) / (vcc / 2 * 606 * 10 ** -5)


def acc_unit_conversion(data, c_min=28000.0, c_max=38000.0):
    """
    Convert raw accelerometer data to m/s2
    
    :param data: Raw accelerometer data
    :param c_min: Minimum calibration value
    :param c_max: Maximum calibration value
    :return: Converted accelerometer data
    """
    return ((data - c_min) / (c_max - c_min)) * 2 - 1


def emg_unit_conversion(data, vcc=3000, resolution=16, gain=1000):
    """
    Convert raw EMG data to millivolts (mV).
    
    :param data: Raw EMG data
    :param vcc: Voltage supply to the sensor
    :param resolution: Sensor resolution in bits
    :param gain: Gain factor of the sensor
    :return: Converted EMG data in mV
    """
    return ((data / (2 ** resolution) - 0.5) * vcc) / gain


def unit_conversion(df):
    """
    Convert raw sensor data to physical units based on sensor type.
    
    :param df: Pandas DataFrame containing raw sensor data
    :return: DataFrame with converted sensor data
    """
    for column in df.columns:
        if 'Electromyogram' in column:
            df[column] = emg_unit_conversion(df[column])
        elif 'Accelerometer' in column:
            df[column] = acc_unit_conversion(df[column])
        elif 'Goniometer' in column:
            df[column] = goniometer_unit_conversion(df[column])
    return df


def get_train_test_data():
    """
    Load the training and testing datasets from CSV files, convert the sensor units,
    and return the processed datasets along with their corresponding labels.
    
    :return: (train_data, y_train, test_data, y_test) - Lists of training/testing data and labels
    """

    def _load_data(data_folder):
        X, y = [], []
        for fl in os.listdir(data_folder):
            df = pd.read_csv(os.path.join(data_folder, fl), names=channels)
            X.append(unit_conversion(df))
            y.append(fl.split('_')[0])  # Extract label from filename
        
        return X, y
    
    # Paths for training and testing folders
    train_folder = os.path.join(dataset_folder, "Training")
    test_folder = os.path.join(dataset_folder, "Testing")

    # Process training data
    train_data, y_train = _load_data(train_folder)

    # Process testing data
    test_data, y_test = _load_data(test_folder)

    return train_data, y_train, test_data, y_test


def plot_random_sample_by_class(train_data, y_train, label, ylim=True):
    """
    Plot a random sensor data sample for a given activity class (label).
    
    :param train_data: List of training data (Pandas DataFrames)
    :param y_train: List of labels corresponding to train_data
    :param label: Activity class to plot (e.g., "walk", "sit")
    :param ylim: Whether to apply Y-axis limits based on data range (default: True)
    :return: DataFrame of the plotted sample and its label
    """
    # Calculate min and max values across all training data for plotting
    max_values = np.max([np.max(d, axis=0) for d in train_data], axis=0)
    min_values = np.min([np.min(d, axis=0) for d in train_data], axis=0)

    # Randomly select an index corresponding to the desired activity class (label)
    idx = np.random.choice(np.where(np.array(y_train) == label)[0])
    df, label = train_data[idx], y_train[idx]

    # Set up subplots for each sensor channel
    fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(5, 10), sharex=True)

    # Plot each channel's data on separate axes
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column], label=column, color="C" + str(i))
        axes[i].legend([column], bbox_to_anchor=(1, 1))
        axes[i].axis("off")  # Hide axis
        if ylim:
            axes[i].set_ylim(max_values[i], min_values[i])  # Apply Y-axis limits if requested
    
    plt.suptitle(label)  # Set the overall plot title
    return df, label


# Plot individual sensor data for a specific sample
def plot_sensor_data(sample, label, sensor_name):
    """
    Plot data from a specific sensor channel for a given sample.
    
    :param sample: Pandas DataFrame containing the sensor data
    :param label: Label of the activity corresponding to the sample
    :param sensor_name: Name of the sensor channel to plot (e.g., "Accelerometer Upper 1")
    """
    plt.plot(sample[sensor_name])
    plt.legend([sensor_name])
    plt.xlabel("Time / ms")
    plt.ylabel(units[sensor_name.split(" ")[0]])
    plt.title(label)
