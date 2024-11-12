import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pywt
import numpy as np


def resample_multichannel_ecg(data, num_target_samples=240):
    """
    对多导联ECG数据应用重采样。
    :param data: numpy array, 原始ECG数据，假设形状为 (samples, leads, points)。
    :param num_target_samples: int, 目标采样点数。
    :return: 重采样后的ECG数据。
    """
    num_leads, num_original_samples = data.shape
    resampled_data = np.zeros((num_leads, num_target_samples))
    for i in range(num_leads):
        resampled_data[i] = resample(data[i], num_target_samples)

    return resampled_data


def denoise_ecg(ecg_signal):
    """
    使用离散小波变换去除ECG信号中的噪声。

    :param ecg_signal: numpy array, 原始ECG信号。
    :return: 去噪后的ECG信号。
    """
    # 指定小波类型和分解级数
    wavelet = 'db6'
    max_level = 9  # 最大分解级数

    # 分解ECG信号
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=max_level)

    # 将特定分量置零（去噪）
    # D1, D2, D3 和 A9 对应 coeffs 的 -1, -2, -3 和第一个元素
    coeffs[-1] = np.zeros_like(coeffs[-1])  # D1
    coeffs[-2] = np.zeros_like(coeffs[-2])  # D2
    coeffs[-3] = np.zeros_like(coeffs[-3])  # D3
    coeffs[0] = np.zeros_like(coeffs[0])  # A9

    # 重构信号
    reconstructed_signal = pywt.waverec(coeffs, wavelet)

    return reconstructed_signal


# 假设 ecg_signal 是你的ECG数据
# denoised_signal = denoise_ecg(ecg_signal)

def preprocess_signals(X_train, X_validation, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def load_labels(label_file):
    import pandas as pd
    labels_df = pd.read_csv(label_file)
    # 创建一个以记录名称为键，标签为值的字典
    labels_dict = labels_df.set_index('Recording')['First_label'].to_dict()
    return labels_dict


def process_ecg_record(ecg_data, target_duration=60, sample_rate=240):
    """
    Process ECG records to a fixed duration of 60 seconds.

    :param ecg_data: numpy array, original ECG data with shape (leads, points).
    :param target_duration: int, target duration in seconds.
    :param sample_rate: int, number of samples per second.
    :return: numpy array, processed ECG data.
    """
    num_leads, current_length = ecg_data.shape
    target_length = target_duration * sample_rate
    processed_data = np.zeros((num_leads, target_length))

    # Check if the data length is already the target length
    if current_length == target_length:
        return ecg_data

    if current_length < target_length:
        # If the record is shorter than the target length, extend it by repeating and overlapping
        extension_factor = int(np.ceil(target_length / current_length))
        extended_data = np.tile(ecg_data, (1, extension_factor))

        # Now cut the extended data to the target length
        # Calculate the overlap size and step
        overlap_size = (extension_factor * current_length - target_length) // (extension_factor - 1)
        step = current_length - overlap_size

        # Initialize starting index
        start_idx = 0

        # Fill the processed_data by overlapping slices
        for idx in range(0, target_length, step):
            end_idx = start_idx + step
            if idx + step > target_length:
                break
            processed_data[:, idx:idx + step] = extended_data[:, start_idx:end_idx]
            start_idx += step

    elif current_length > target_length:
        # If the record is longer than the target length, truncate it
        processed_data = ecg_data[:, :target_length]

    return processed_data


def cross_validation_splits(data, labels, n_splits=5):
    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    skf_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  # For train-validation split
    x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []  # List to hold all splits
    for train_index, test_index in skf_outer.split(data, labels):
        X_train_test, X_test = data[train_index], data[test_index]
        Y_train_test, Y_test = labels[train_index], labels[test_index]

        # Split train_test data to create a validation set
        train_index_inner, val_index_inner = next(skf_inner.split(X_train_test, Y_train_test))
        X_train, X_val = X_train_test[train_index_inner], X_train_test[val_index_inner]
        Y_train, Y_val = Y_train_test[train_index_inner], Y_train_test[val_index_inner]
        x_train.append(X_train)
        x_val.append(X_val)
        x_test.append(X_test)
        y_train.append(Y_train)
        y_val.append(Y_val)
        y_test.append(Y_test)
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    return x_train, x_val, x_test, y_train, y_val, y_test


def resample_ecg(data, original_freq, target_freq, duration):
    # num_samples = int(target_freq / original_freq * data.shape[1])
    num_samples = int(data.shape[1] // 5)
    resampled_data = resample(data, num_samples, axis=1)
    return resampled_data


# Standardize length of ECG recordings to 10 seconds
def standardize_length(data, target_length, sampling_rate):
    standardized_data = []
    for record in data:
        if len(record) > target_length * sampling_rate:
            # Crop
            standardized_data.append(record[:target_length * sampling_rate])
        elif len(record) < target_length * sampling_rate:
            # Repeat the signal to fill 10 seconds
            repeat_times = (target_length * sampling_rate + len(record) - 1) // len(record)
            repeated_signal = np.tile(record, repeat_times)[:target_length * sampling_rate]
            standardized_data.append(repeated_signal)
        else:
            standardized_data.append(record)
    return np.array(standardized_data)


def load_and_process_ecg_dataset(dataset_folders, labels_dict, target_length=30000):
    all_data = []
    all_labels = []

    for folder in dataset_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.mat'):
                record_id = filename.replace('.mat', '')
                if record_id in labels_dict:
                    file_path = os.path.join(folder, filename)
                    mat_data = sio.loadmat(file_path)
                    if 'ECG' in mat_data and 'data' in mat_data['ECG'].dtype.names:
                        ecg_data = mat_data['ECG']['data'][0, 0]
                        ecg_data = resample_ecg(ecg_data, 500, 100, 10)
                        ecg_data = standardize_length(ecg_data, 10, 100)
                        # processed_ecg = process_ecg_record(ecg_data)
                        all_data.append(ecg_data)
                        all_labels.append(labels_dict[record_id])

    return np.array(all_data), np.array(all_labels)
    # Load labels


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        super(ECGDataset, self).__init__()
        self.data = signals
        self.label = labels
        self.num_classes = self.label.shape[1]
        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        x = x.transpose()
        x = torch.tensor(x.copy(), dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return x, y

    def __len__(self):
        return len(self.data)


def save_data():
    label_file = 'F:/dataset/2018/cpsc_database.csv'
    labels_dict = load_labels(label_file)
    # Specify dataset folders
    dataset_folders = ['F:/dataset/2018/TrainingSet1', 'F:/dataset/2018/TrainingSet2', 'F:/dataset/2018/TrainingSet3']
    # Load and process the dataset
    data, labels = load_and_process_ecg_dataset(dataset_folders, labels_dict)
    np.save('data2.npy', data)
    np.save('labels2.npy', labels)


def load_data(batch_size):
    # Split the data using 5-fold cross-validation
    data, labels = np.load('data2.npy'), np.load('labels2.npy')
    labels = [list(set([label])) for label in labels]
    data_num = len(data)
    shuffle_ix = np.random.permutation(np.arange(data_num))
    mlb = MultiLabelBinarizer(classes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    labels = mlb.fit_transform(labels)
    data = data[shuffle_ix]
    labels = labels[shuffle_ix]
    x_train = data[:int(data_num * 0.8)]
    x_val = data[int(data_num * 0.8):int(data_num * 0.9)]
    x_test = data[int(data_num * 0.9):]
    x_train, x_val, x_test = preprocess_signals(x_train, x_val, x_test)
    y_train = labels[:int(data_num * 0.8)]
    y_val = labels[int(data_num * 0.8):int(data_num * 0.9)]
    y_test = labels[int(data_num * 0.9):]
    # x_train, x_val, x_test, y_train, y_val, y_test = cross_validation_splits(data, labels)
    train_loader = DataLoader(dataset=ECGDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=ECGDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=ECGDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    load_data(32)
