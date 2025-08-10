
import os
import gc
import glob
import mne
import numpy as np
import pandas as pd
from scipy.signal import stft
import subprocess
import yaml
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from src.performance_monitor import PerformanceMonitor

@dataclass
class PipelineConfig:
    # Filtering parameters
    filter_l_freq: float = 0.5
    filter_h_freq: float = 35.0

    # ICA parameters
    ica_n_components: int = 2
    ica_random_state: int = 97
    ica_fit_l_freq: float = 1.0
    ica_eog_threshold: float = 2.0

    # Epoching parameters
    epoch_duration_secs: int = 30

    # STFT Feature Engineering parameters
    stft_nperseg: int = 256
    stft_noverlap: int = 128

    # Visual Validation
    plot_ica_diagnostics: bool = True
    plot_save_dir: str = "../reports/figures/ica_diagnostics"


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
            all_psg_files = sorted(glob.glob(os.path.join(self.raw_data_dir, '**', '*-PSG.edf'), recursive=True))
            all_hyp_files = sorted(glob.glob(os.path.join(self.raw_data_dir, '**', '*-Hypnogram.edf'), recursive=True))

            print(f"[_find_edf_files] Found {len(all_psg_files)} PSG files and {len(all_hyp_files)} Hypnogram files in total.")

            if not all_psg_files or not all_hyp_files:
                print(f"[_find_edf_files] WARNING: No data files found in {self.raw_data_dir}.")
                return []

            hyp_map = {os.path.basename(f)[:7]: f for f in all_hyp_files}

            paired_files = []
            for psg_path in all_psg_files:
                psg_key = os.path.basename(psg_path)[:7]
                hyp_path = hyp_map.get(psg_key)
                if hyp_path:
                    subject_id = os.path.basename(psg_path).replace("-PSG.edf", "")
                    paired_files.append({'psg': psg_path, 'hypnogram': hyp_path, 'id': subject_id})
                else:
                    print(f"[_find_edf_files] WARNING: No matching Hypnogram file found for {psg_path}.")

            return paired_files

        except Exception as e:
            print(f"[_find_edf_files] WARNING: Failed to find EDF files: {e}")
            return []

    # def _load_labels(self, hypnogram_file: str) -> tuple:
    #     try:
    #         annotations = mne.read_annotations(hypnogram_file)
    #         label_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    #                     'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
    #         labels = np.array([label_map.get(desc, -1) for desc in annotations.description])
    #         return labels, annotations.onset, annotations.duration
    #     except Exception as e:
    #         print(f"[_load_labels] WARNING: Failed to load labels for {hypnogram_file}: {e}")

    def _filter_signal(self, psg_file: str, config: PipelineConfig) -> mne.io.Raw:
        print("  - Step 1: Loading and Filtering...")
        try:
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=True)
            print("[_filter_signal] Resetting filter info from EDF header.")
            with raw.info._unlock():
                raw.info['highpass'] = 0.
                raw.info['lowpass'] = raw.info['sfreq'] / 2.0

            required_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']

            if not all(ch in raw.ch_names for ch in required_channels):
                print(f"[_filter_signal] WARNING: Missing one of the required channels in {psg_file}. Skipping subject.")
                return None

            raw.pick(required_channels)
            raw.rename_channels({
                'EEG Fpz-Cz': 'EEG1',       # Primary EEG channel
                'EEG Pz-Oz': 'EEG2',        # Secondary EEG channel for ICA
                'EOG horizontal': 'EOG'     # EOG channel
            })
            raw.set_channel_types({'EEG1': 'eeg', 'EEG2': 'eeg', 'EOG': 'eog'})

            # print("[_filter_signal] Applying Notch filter (50Hz)...")
            # raw.notch_filter(50, verbose=True)

            print(f"[_filter_signal] Applying Band-pass filter ({config.filter_l_freq}-{config.filter_h_freq}Hz)...")
            raw.filter(l_freq=config.filter_l_freq, h_freq=config.filter_h_freq, verbose=True)

            # try:
            #     print("    - Attempting filter order: Notch -> Band-pass")
            #     raw.notch_filter(50, verbose=True)
            #     raw.filter(0.5, 35, verbose=True)
            # except ValueError as e:
            #     print(f"    - WARNING: Initial filter order failed ({e}). Trying fallback: Band-pass -> Notch")
            #     raw.filter(0.5, 35, verbose=True)
            #     raw.notch_filter(50, verbose=True)

            print("[_filter_signal] Filtering successful.")
            return raw
        except KeyError as e:
            print(f"[_filter_signal] WARNING: Could not find required channels (e.g., 'EEG Fpz-Cz') in {psg_file}. Skipping this subject.")
            return None
        except Exception as e:
            print(f"[_filter_signal] WARNING: A critical error occurred during filtering for {psg_file}: {e}")
            return None

    def _remove_artifacts_ica(self, raw: mne.io.Raw, subject_id: str, config: PipelineConfig) -> mne.io.Raw:
        print("  - Step 2: Running ICA for artifact removal...")
        try:
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            if len(eeg_picks) < 2:
                print(f"[_remove_artifacts_ica] WARNING: Not enough EEG channels ({len(eeg_picks)}) to run ICA. Skipping artifact removal.")
                return raw

            ica = mne.preprocessing.ICA(
                n_components=config.ica_n_components,
                random_state=config.ica_random_state,
                max_iter='auto',
                verbose=True
            )
            ica.fit(raw.copy().filter(l_freq=config.ica_fit_l_freq, h_freq=None, verbose=True), picks=eeg_picks, verbose=True)
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG', threshold=config.ica_eog_threshold, verbose=True)

            if eog_indices:
                print(f"[_remove_artifacts_ica] ICA found EOG component(s): {eog_indices}. Excluding them...")
                ica.exclude = eog_indices
                if config.plot_ica_diagnostics:
                    self._plot_ica_results(ica, raw, eog_indices, subject_id, config)
                ica.apply(raw, verbose=True)
            else:
                print("[_remove_artifacts_ica] ICA did not find any EOG components.")
        except Exception as e:
            print(f"[_remove_artifacts_ica] WARNING: ICA failed for {subject_id}: {e} — continuing with uncleaned signal.")
        return raw

    def _plot_ica_results(self, ica: mne.preprocessing.ICA, raw: mne.io.Raw, eog_indices: list, subject_id: str, config: PipelineConfig):
        """Saves a diagnostic plot of the ICA results."""
        print(f"[_plot_ica_results] Generating ICA diagnostic plot for {subject_id}...")
        try:
            os.makedirs(config.plot_save_dir, exist_ok=True)

            fig = ica.plot_sources(raw, show=False, title=f"ICA Sources for {subject_id}")
            plt.savefig(os.path.join(config.plot_save_dir, f"{subject_id}_ica_sources.png"))
            plt.close(fig)

            overlay_fig = ica.plot_overlay(raw, exclude=eog_indices, show=False, title=f"Raw vs. Cleaned EEG for {subject_id}")
            plt.savefig(os.path.join(config.plot_save_dir, f"{subject_id}_ica_overlay.png"))
            plt.close(overlay_fig)

        except Exception as e:
            print(f"[_plot_ica_results] WARNING: Failed to generate plots for {subject_id}: {e}")

    def _create_epochs_and_features(self, cleaned_raw: mne.io.Raw, hypnogram_file: str, config: PipelineConfig) -> tuple:
        print("  - Step 3: Epoching and creating spectrograms STFT...")
        try:
            annotations = mne.read_annotations(hypnogram_file)
            # annotations.crop(annotations[0]['onset'], annotations[-1]['onset'], use_orig_time=False)
            cleaned_raw.set_annotations(annotations, emit_warning=True)

            event_id_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                            'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}

            events, _ = mne.events_from_annotations(
                cleaned_raw, event_id=event_id_map,
                chunk_duration=config.epoch_duration_secs
            )


            if len(events) == 0:
                print(f"[_create_epochs_and_features] No valid sleep stages found in {hypnogram_file}. Skipping subject.")
                return None, None

            tmax = config.epoch_duration_secs - 1. / cleaned_raw.info['sfreq']

            epochs = mne.Epochs(
                raw=cleaned_raw, events=events, event_id=event_id_map,
                tmin=0., tmax=tmax, baseline=None, preload=True, verbose=True
            )

            # events = np.array([onsets * cleaned_raw.info['sfreq'], durations * cleaned_raw.info['sfreq'], labels]).T.astype(int)
            # valid_mask = events[:, 2] >= 0
            # epochs = mne.Epochs(cleaned_raw, events[valid_mask], tmin=0., tmax=30., baseline=None, preload=True, verbose=True)

            epoch_labels = epochs.events[:, 2]
            print("[_create_epochs_and_features] Selecting 'EEG1' channel for spectrogram creation.")
            epochs_data_eeg = epochs.get_data(picks=['EEG1'])
            print(f"[_create_epochs_and_features] Created {len(epochs_data_eeg)} epochs.")

            sfreq = cleaned_raw.info['sfreq']
            _, _, Sxx = stft(
                epochs_data_eeg, fs=sfreq,
                nperseg=config.stft_nperseg,
                noverlap=config.stft_noverlap
            )
            print(f"[_create_epochs_and_features] Created spectrograms with shape: {Sxx.shape}")

            return np.abs(Sxx).astype(np.float32), epoch_labels
        except Exception as e:
            print(f"[_create_epochs_and_features] WARNING: Sub-Process failed: {e}")
            return None, None
        # finally:
        #     del epochs, cleaned_raw, events, event_id
        #     gc.collect()

    def _normalize_and_save(self, spectrograms: np.ndarray, labels: np.ndarray, subject_id: str):
        print("  - Step 4: Normalizing and saving...")
        try:
            if not os.path.exists(self.processed_data_dir):
                os.makedirs(self.processed_data_dir)

            db_spectrograms = 10 * np.log10(spectrograms + 1e-10)
            mean = np.mean(db_spectrograms, keepdims=True)
            std = np.std(db_spectrograms, keepdims=True)
            normalized_spectrograms = (db_spectrograms - mean) / (std + 1e-9)

            save_path = os.path.join(self.processed_data_dir, f"{subject_id}_processed.npz")
            np.savez_compressed(save_path, x=normalized_spectrograms.astype(np.float32), y=labels)
            print(f"[_normalize_and_save] Saved processed data for {subject_id} to {save_path}")
        except Exception as e:
            print(f"[_normalize_and_save] WARNING: Sub-Process failed: {e}")

    def _process_single_subject(self, files: dict, config: PipelineConfig):
        subject_id = files['id']
        print(f"\n--- Processing Subject: {subject_id} ---")
        raw_filtered, cleaned_raw, spectrograms, epoch_labels = None, None, None, None
        try:
            raw_filtered = self._filter_signal(files['psg'], config)
            if raw_filtered is None: return

            cleaned_raw = self._remove_artifacts_ica(raw_filtered, subject_id, config)
            if cleaned_raw is None: return

            spectrograms, epoch_labels = self._create_epochs_and_features(cleaned_raw, files['hypnogram'], config)
            if spectrograms is None or epoch_labels is None: return

            self._normalize_and_save(spectrograms, epoch_labels, subject_id)

            print(f"[_process_single_subject] SUCCESS: Completed processing for {subject_id}.")
        except Exception as e:
            print(f"[_process_single_subject] FAILED to process subject {subject_id}: {e}")
        finally:
            del raw_filtered, cleaned_raw, spectrograms, epoch_labels
            gc.collect()

    def run_preprocessing(self, config: PipelineConfig, num_subjects: int = None):
        if not self.subject_files:
            print("No subjects found to process. Halting execution.")
            return

        subjects_to_process = self.subject_files[:num_subjects] if num_subjects else self.subject_files

        self.monitor.start()

        for i, files in enumerate(subjects_to_process):
            subject_id = files['id']
            print(f"\n[run_preprocessing] Starting subject {i+1}/{len(subjects_to_process)} ({subject_id})...")
            try:
                self._process_single_subject(files, config)
            except Exception as e:
                print(f"[run_preprocessing] FAILED to process subject {subject_id}: {e}")

        self.monitor.stop()
        self.monitor.generate_html_report(task_name=f"DataPreprocessor_{len(subjects_to_process)}_subjects")
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
