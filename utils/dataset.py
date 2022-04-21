import numpy as np
import torch
import utils.utils as utils
from torch.utils.data import Dataset, Sampler


class MMFit(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """
    def __init__(self, modality_filepaths, label_path, window_length, skeleton_window_length, sensor_window_length,
                 skeleton_transform, sensor_transform):
        """
        Initialize MMFit Dataset object.
        :param modality_filepaths: Modality - file path mapping (dict) for a workout.
        :param label_path: File path to MM-Fit CSV label file for a workout.
        :param window_length: Window length in seconds.
        :param skeleton_window_length: Skeleton window length in number of samples.
        :param sensor_window_length: Sensor window length in number of samples.
        :param skeleton_transform: Transformation functions to apply to skeleton data.
        :param sensor_transform: Transformation functions to apply to sensor data.
        """
        self.window_length = window_length
        self.skeleton_window_length = skeleton_window_length
        self.sensor_window_length = sensor_window_length
        self.skeleton_transform = skeleton_transform
        self.sensor_transform = sensor_transform
        self.modalities = {}
        for modality, filepath in modality_filepaths.items():
            self.modalities[modality] = utils.load_modality(filepath)

        self.ACTIONS = {'squats': 0, 'lunges': 1, 'bicep_curls': 2, 'situps': 3, 'pushups': 4, 'tricep_extensions': 5,
                        'dumbbell_rows': 6, 'jumping_jacks': 7, 'dumbbell_shoulder_press': 8,
                        'lateral_shoulder_raises': 9, 'non_activity': 10}
        self.labels = utils.load_labels(label_path)

    def __len__(self):
        return self.modalities['pose_3d'].shape[1] - self.skeleton_window_length - 30
        
    def __getitem__(self, i):
        frame = self.modalities['pose_3d'][0, i, 0]
        sample_modalities = {}
        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame > (row[0] - self.skeleton_window_length/2)) and (frame < (row[1] - self.skeleton_window_length/2)):
                label = row[3]
                reps = row[2]
                break

        for modality, data in self.modalities.items():
            if data is None:
                if 'pose_2d' in modality:
                    sample_modalities[modality] = torch.zeros(2, self.skeleton_window_length, 17)
                elif 'pose_3d' in modality:
                    sample_modalities[modality] = torch.zeros(3, self.skeleton_window_length, 16)
                elif 'hr' in modality:
                    sample_modalities[modality] = torch.zeros(1, self.sensor_window_length)
                else:
                    sample_modalities[modality] = torch.zeros(3, self.sensor_window_length)
            else:
                if 'pose' in modality:
                    sample_modalities[modality] = torch.as_tensor(self.skeleton_transform(
                        data[:, i:i+self.skeleton_window_length, 1:]), dtype=torch.float)
                else:
                    start_frame_idx = np.searchsorted(data[:, 0], frame, 'left')

                    time_interval_s = (data[(start_frame_idx + 1):, 1] - data[start_frame_idx, 1]) / 1000
                    end_frame_idx = np.searchsorted(time_interval_s, self.window_length, 'left') + start_frame_idx + 1
                    if end_frame_idx >= data.shape[0]:
                        raise Exception('Error: end_frame_idx, {}, is out of index for data array with length {}'.
                                        format(end_frame_idx, data.shape[0]))

                    if 'hr' in modality:
                        sample_modalities[modality] = torch.as_tensor(self.sensor_transform(
                            data[start_frame_idx:end_frame_idx, 2].T), dtype=torch.float)
                    else:
                        sample_modalities[modality] = torch.as_tensor(self.sensor_transform(
                            data[start_frame_idx:end_frame_idx, 2:].T), dtype=torch.float)
        
        return sample_modalities, self.ACTIONS[label], reps


class SequentialStridedSampler(Sampler):
    """
    PyTorch Sampler Class to sample elements sequentially using a specified stride, always in the same order.
    Arguments:
        data_source (Dataset):
        stride (int):
    """

    def __init__(self, data_source, stride):
        """
        Initialize SequentialStridedSampler object.
        :param data_source: Dataset to sample from.
        :param stride: Stride to slide window in seconds.
        """
        self.data_source = data_source
        self.stride = stride
    
    def __len__(self):
        return len(range(0, len(self.data_source), self.stride))

    def __iter__(self):
        return iter(range(0, len(self.data_source), self.stride))
