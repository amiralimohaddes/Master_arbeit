
import os
from typing import Dict, List, Tuple
import librosa
import numpy as np

def get_machine_types(dataset_root: str) -> List[str]:
    """Return a list of machine types (folder names) in the dataset."""
    machine_types = []
    for entry in os.listdir(dataset_root):
        full_path = os.path.join(dataset_root, entry)
        if os.path.isdir(full_path):
            machine_types.append(entry)
    return machine_types


def load_audio_files(file_list: List[str], sr: int = 16000) -> List[np.ndarray]:
    """Load audio files as numpy arrays."""
    audio_data = []
    for f in file_list:
        try:
            y, _ = librosa.load(f, sr=sr, mono=True)
            audio_data.append(y)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            audio_data.append(None)
    return audio_data

def load_data(dataset_root: str, sr: int = 16000) -> Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]]:
    """
    Loads train and test audio data for each machine type in the attached 'dataset' folder.
    Only normal data is used for training. Both normal and anomaly for testing.
    Returns a dict: {machine_type: {'train': [(path, audio)], 'test': [(path, audio)]}}
    """
    data = {}
    machine_types = get_machine_types(dataset_root)
    for machine in machine_types:
        machine_path = os.path.join(dataset_root, machine)
        train_dir = os.path.join(machine_path, 'train')
        test_dir = os.path.join(machine_path, 'test')
        train_files = []
        test_files = []
        # Training: only normal files
        if os.path.isdir(train_dir):
            for fname in os.listdir(train_dir):
                if 'normal' in fname and fname.endswith('.wav'):
                    train_files.append(os.path.join(train_dir, fname))
        # Testing: all files (normal and anomaly)
        if os.path.isdir(test_dir):
            for fname in os.listdir(test_dir):
                if fname.endswith('.wav'):
                    test_files.append(os.path.join(test_dir, fname))
        train_audio = load_audio_files(sorted(train_files), sr=sr)
        test_audio = load_audio_files(sorted(test_files), sr=sr)
        data[machine] = {
            'train': list(zip(sorted(train_files), train_audio)),
            'test': list(zip(sorted(test_files), test_audio))
        }
    return data

# Example usage:
# dataset_root = r"./dataset"  # Use the attached dataset folder
# data = load_data(dataset_root)
# print(data["fan"]["train"][0])  # (filepath, audio array)
