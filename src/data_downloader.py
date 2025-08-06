
import os
import subprocess
import shutil

class DatasetDownloader:
    """
    Handles the download and extraction of the Sleep-EDF Expanded dataset from PhysioNet.

    Attributes:
        base_url (str): The base URL for the dataset files.
        download_dir (str): The local directory to save the dataset.
    """
    def __init__(self, base_url: str, download_dir: str = './data/raw_edf'):
        """
        Initializes the downloader with the dataset URL and target directory.

        Args:
            base_url (str): The URL from which to download the dataset.
            download_dir (str): The path to the directory where data will be stored.
                                Defaults to './data/raw_edf'.
        """
        self.base_url = base_url
        self.download_dir = download_dir
        print(f"[DatasetDownloader] Downloader initialized. Data will be saved to: {self.download_dir}")

    def _flatten_directory(self):
        """Checks if wget is installed and available in the system's PATH."""
        source_dir = os.path.join(self.download_dir, 'physionet.org/files/sleep-edfx/1.0.0')
        if not os.path.exists(source_dir):
            print(f"[_flatten_directory] Source directory '{source_dir}' not found. No cleanup needed.")
            return

        print(f"[_flatten_directory] Cleaning up directory: moving files from '{source_dir}'...")
        for item_name in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item_name)
            dest_item = os.path.join(self.download_dir, item_name)
            shutil.move(source_item, dest_item)

        print("[_flatten_directory] File move complete.")

        cleanup_dir = os.path.join(self.download_dir, 'physionet.org')
        if os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir)
        print("[_flatten_directory] Directory cleanup complete.")

    def download(self):
        """
        Downloads the entire Sleep-EDF Expanded dataset using wget.

        This method uses a recursive wget command to fetch all files from the
        PhysioNet directory.
        """
        if not os.path.exists(self.download_dir):
            print(f"[download] Creating directory: {self.download_dir}")
            os.makedirs(self.download_dir)
        else:
            print(f"[download] Directory already exists: {self.download_dir}")

        print("\n[download] Starting the download of the Sleep-EDF Expanded dataset. This may take a long time...")

        # Construct the wget command
        # -r: recursive
        # -N: don't re-retrieve files unless newer than local
        # -c: continue getting a partially-downloaded file
        # -np: no-parent, don't ascend to the parent directory
        # -P: specify the directory prefix to save files to
        # --reject: comma-separated list of file names to reject
        command = [
            'wget',
            '-r', '-N', '-c', '-np',
            f'--directory-prefix={self.download_dir}',
            '--reject="index.html*"',
            self.base_url
        ]

        try:
            print(f"[download] Executing command: {' '.join(command)}")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

            rc = process.poll()
            if rc == 0:
                print("\n[download] Download completed successfully!")
                self._flatten_directory()

            else:
                print(f"\n[download] Download failed with return code: {rc}")
        except KeyboardInterrupt:
            print("\n[download] Download interrupted by the user.")
        except subprocess.CalledProcessError as e:
            print(f"[download] Download failed with error, stderr and return code: {e} [{e.stderr}] [{e.returncode}]")
        except FileNotFoundError:
            print("[download] Error: 'wget' command failed. Ensure it is installed and in your PATH.")
        except Exception as e:
            print(f"[download] An unexpected error occurred: {e}")

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
