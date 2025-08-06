
import os
import gc
import glob
import mne
import numpy as np
import pandas as pd
from scipy.signal import stft
import subprocess
from src.performance_monitor import PerformanceMonitor

class DataPreprocessor:
    """
    Handles the comprehensive, end-to-end preprocessing of the Sleep-EDF dataset.
    
    This class is aggressively optimized for memory efficiency to handle large files
    in environments like Google Colab. It processes data one subject at a time,
    explicitly manages memory, and implements advanced techniques like ICA.
    """
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str, monitor: PerformanceMonitor):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.monitor = monitor
        self.subject_files = self._find_edf_files()
        os.makedirs(self.processed_data_dir, exist_ok=True)
        print(f"[DataPreprocessor] Ready. Found {len(self.subject_files)} subjects.")

    def _find_edf_files(self) -> list:
        try:
            psg_files = []
            hyp_files = []
            subdirectories = ['sleep-cassette', 'sleep-telemetry']

            print(f"[_find_edf_files] Searching for EDF files in: {subdirectories}")
            
            for subdir in subdirectories:
                subdir_path = os.path.join(self.raw_data_dir, subdir)
                print(f"[_find_edf_files] Searching for EDF files in: {subdir_path}")
                if os.path.exists(subdir_path):
                    psg_files.extend(sorted(glob.glob(os.path.join(subdir_path, '**', '*-PSG.edf'), recursive=True)))
                    hyp_files.extend(sorted(glob.glob(os.path.join(subdir_path, '**', '*-Hypnogram.edf'), recursive=True)))
                else:
                    print(f"[_find_edf_files] Directory not found: {subdir_path}")
                    return []

            if not psg_files or not hyp_files:
                print(f"[_find_edf_files] WARNING: No PSG or Hypnogram files were found. Please check the RAW_DATA_DIR path and ensure the subdirectories exist and are synced.")
                return []

            print(f"[_find_edf_files] Found {len(psg_files)} PSG files and {len(hyp_files)} Hypnogram files.")

            hyp_map = { os.path.basename(f).replace('-Hypnogram.edf', ''): f for f in hyp_files }
            paired_files = []
            for psg_path in psg_files:
                psg_id = os.path.basename(psg_path).replace('-PSG.edf', '')
                hyp_path = hyp_map.get(psg_id)

                if hyp_path:
                    paired_files.append({'psg': psg_path, 'hypnogram': hyp_path, 'id': psg_id})
                else:
                    print(f"[_find_edf_files] WARNING: No matching hypnogram found for {psg_path}")

            return paired_files

        except Exception as e:
            print(f"[_find_edf_files] WARNING: Failed to find EDF files: {e}")
            return []

    def _load_labels(self, hypnogram_file: str) -> tuple:
        try:
            annotations = mne.read_annotations(hypnogram_file)
            label_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                        'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
            labels = np.array([label_map.get(desc, -1) for desc in annotations.description])
            return labels, annotations.onset, annotations.duration
        except Exception as e:
            print(f"[_load_labels] WARNING: Failed to load labels for {hypnogram_file}: {e}")

    def _filter_signal(self, psg_file: str) -> mne.io.Raw:
        print("  - Step 1: Loading and Filtering...")
        try:
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=True)
            raw.rename_channels({'EEG Fpz-Cz': 'EEG', 'EOG horizontal': 'EOG horizontal'})
            raw.set_channel_types({'EOG horizontal': 'eog'})
            raw.pick_channels(['EEG', 'EOG horizontal'])
            raw.notch_filter(50, verbose=False)
            raw.filter(0.5, 35, verbose=False)
            return raw
        except Exception as e:
            print(f"[_filter_signal] WARNING: Filter failed for {psg_file}: {e}")

    def _remove_artifacts_ica(self, raw: mne.io.Raw, subject_id: str) -> mne.io.Raw:
        print("  - Step 2: Running ICA for artifact removal...")
        try:
            ica = mne.preprocessing.ICA(n_components=2, random_state=97, max_iter=800, verbose=True)
            ica.fit(raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False), verbose=False)
            eog_indices, _ = ica.find_bads_eog(raw, ch_name='EOG horizontal', verbose=False)
            if eog_indices:
                ica.exclude = eog_indices
                ica.apply(raw, verbose=False)
        except Exception as e:
            print(f"[_remove_artifacts_ica] WARNING: ICA failed for {subject_id}: {e} — continuing with uncleaned signal.")
        return raw

    def _create_epochs_and_features(self, cleaned_raw: mne.io.Raw, labels, onsets, durations) -> tuple:
        print("  - Step 3: Epoching and creating spectrograms STFT...")
        try:
            events = np.array([onsets * cleaned_raw.info['sfreq'], durations * cleaned_raw.info['sfreq'], labels]).T.astype(int)
            valid_mask = events[:, 2] >= 0
            epochs = mne.Epochs(cleaned_raw, events[valid_mask], tmin=0., tmax=30., baseline=None, preload=True, verbose=True)
            epoch_labels = epochs.events[:, 2]
            
            epochs_data_eeg = epochs.get_data(picks=['EEG'])
            del epochs, cleaned_raw; gc.collect()

            sfreq, nperseg, noverlap = 100, 256, 128
            _, _, Sxx = stft(epochs_data_eeg, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            return np.abs(Sxx).astype(np.float32), epoch_labels
        except Exception as e:
            print(f"[_create_epochs_and_features] WARNING: Sub-Process failed: {e}")

    def _normalize_and_save(self, spectrograms: np.ndarray, labels: np.ndarray, subject_id: str):
        print("  - Step 4: Normalizing and saving...")
        try:
            if not os.path.exists(self.processed_data_dir):
                os.makedirs(self.processed_data_dir)
            else:
                db_spectrograms = 10 * np.log10(spectrograms + 1e-10)
                mean, std = np.mean(db_spectrograms), np.std(db_spectrograms)
                normalized_spectrograms = (db_spectrograms - mean) / std
                
                save_path = os.path.join(self.processed_data_dir, f"{subject_id}_processed.npz")
                np.savez_compressed(save_path, x=normalized_spectrograms.astype(np.float32), y=labels)
                print(f"[_normalize_and_save] Saved processed data for {subject_id} to {save_path}")
        except Exception as e:
            print(f"[_normalize_and_save] WARNING: Sub-Process failed: {e}")

    def _process_single_subject(self, files: dict):
        subject_id = os.path.basename(files['psg']).split('-')[0]
        print(f"\n--- Processing Subject: {subject_id} ---")
        try:
            labels, onsets, durations = self._load_labels(files['hypnogram'])
            raw_filtered = self._filter_signal(files['psg'])
            cleaned_raw = self._remove_artifacts_ica(raw_filtered, subject_id)
            spectrograms, epoch_labels = self._create_epochs_and_features(cleaned_raw, labels, onsets, durations)
            self._normalize_and_save(spectrograms, epoch_labels, subject_id)
            print(f"[_process_single_subject] Completed processing for {subject_id}.")
        except Exception as e:
            print(f"[_process_single_subject] FAILED to process subject {subject_id}: {e}")
        finally:
            del raw_filtered, cleaned_raw, spectrograms, epoch_labels
            gc.collect()

    def run_preprocessing(self, num_subjects: int = None):
        if self.subject_files <= 0:
            print("No subjects found to process. Halting execution.")
            return

        subjects_to_process = self.subject_files[:num_subjects] if num_subjects else self.subject_files
        
        for i, files in enumerate(subjects_to_process):
            print(f"\n[run_preprocessing] Starting subject {i+1}/{len(subjects_to_process)}...")
            self.monitor.start()
            try:
                self._process_single_subject(files)
            except Exception as e:
                print(f"[run_preprocessing] FAILED to process subject {os.path.basename(files['psg'])}: {e}")
            finally:
                self.monitor.stop()
                subject_id = os.path.basename(files['psg']).split('-')[0]
                self.monitor.generate_html_report(task_name=f"preprocess_{subject_id}")
        print("\n--- All selected subjects have been processed. ---")

    def push_script_to_github(self, file_path: str, commit_message: str, git_token: str, git_username: str, git_repo: str):
        """
        Adds, commits, and pushes a specific file to the GitHub repository.

        Args:
            file_path (str): The path to the file to be pushed (e.g., 'src/data_downloader.py').
            commit_message (str): The message for the git commit.
            git_token (str): Your GitHub Personal Access Token.
            git_username (str): Your GitHub username.
            git_repo (str): Your GitHub repository name.
        """
        print(f"\n[push_script_to_github] Attempting to push '{file_path}' to GitHub...")

        try:
            subprocess.run(['git', 'add', file_path, '.gitignore'], check=True)

            commit_process = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True)
            if "nothing to commit" in commit_process.stdout or "no changes added" in commit_process.stdout:
                 print("[push_script_to_github] No new changes to commit.")
                 return
            elif commit_process.returncode != 0:
                 print("[push_script_to_github] Git commit failed:", commit_process.returncode)
                 return

            print("[push_script_to_github] Commit successful.")

            remote_url = f"https://{git_token}@github.com/{git_username}/{git_repo}.git"
            subprocess.run(['git', 'push', remote_url], check=True)

            print(f"[push_script_to_github] Successfully pushed '{file_path}' to GitHub.")

        except KeyboardInterrupt:
            print("\n[push_script_to_github] Push interrupted by the user.")
        except subprocess.CalledProcessError as e:
            print(f"[push_script_to_github] Push failed with error, stderr and return code: {e} [{e.stderr}] [{e.returncode}]")
        except FileNotFoundError:
            print("[push_script_to_github] Error: 'git' command failed. Ensure it is installed and in your PATH.")
        except Exception as e:
            print(f"[push_script_to_github] An unexpected error occurred: {e}")
