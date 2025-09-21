import os
import subprocess

def sync_log_folders(base_path):
    """
    Syncs all subdirectories in the base_path to W&B using `wandb sync`.
    
    Args:
    - base_path (str): The path to the directory containing log folders.
    """
    # List all items in base_path
    all_items = os.listdir(base_path)

    # Filter to get only directories
    folders = [item for item in all_items if os.path.isdir(os.path.join(base_path, item))and 
               ("20241221" in item)]

    # Iterate through each folder and run the wandb sync command
    for i,folder in enumerate(folders):
        folder_path = os.path.join(base_path, folder)
        
        # Check if the item is a directory
        if os.path.isdir(folder_path):
            print(f"{i} Syncing {folder_path}...")
            subprocess.run(["wandb", "sync", folder_path])
        else:
            print(f"Skipped {folder_path}, not a directory.")
if __name__ == "__main__":
    # Specify the base path containing the folders to be synced
    base_path = "./wandb"

    # Call the function to start syncing
    sync_log_folders(base_path)
