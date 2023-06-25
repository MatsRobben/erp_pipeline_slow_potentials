import matplotlib.pyplot as plt
import os

def create_folders(folder_list, new_dir):
    """
    Create folders for each item in the folder list.

    Args:
        folder_list (list): List of folder names.
        new_dir (str): Path to the parent directory.

    Returns:
        list: List of created folder paths.
    """
    paths = []
    for folder in folder_list:
        path = f"{new_dir}/{folder}"
        make_dir(path)
        paths.append(path)
    return paths

def make_dir(path):
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_plot(fig, folder, file_name, file_extension, par=None):
    """
    Save a matplotlib figure to disk.

    Args:
        fig (matplotlib.figure.Figure): Figure object to save.
        folder (str): Folder name for saving the file.
        file_name (str): Name of the file.
        file_extension (str): File extension (e.g., '.png', '.jpg').
        par (str, optional): Participant name or identifier.

    Note:
        If `par` is provided, the file will be saved in the following format:
        'saved-data/{par}/{folder}/{file_name}{file_extension}'
        Otherwise, it will be saved as '{folder}/{file_name}{file_extension}'.
    """
    if par:
        path = f"saved-data/{par}/{folder}"
    else:
        path = f"{folder}"
        
    make_dir(path)
    fig.savefig(f"{path}/{file_name}{file_extension}")
    plt.close(fig)

def clear_figs():
    """
    Clear all matplotlib figures.
    """
    all_fig_nums = plt.get_fignums()
    for fig_num in all_fig_nums:
        fig = plt.figure(fig_num)
        fig.clear()
        plt.close(fig)

def get_participants():
    """
    Get a list of participant names.

    Returns:
        list: List of participant names.
    """
    local_path = os.path.dirname(os.path.realpath('main.py'))
    local_data_path = f"{local_path}\Data"

    participants = os.listdir(local_data_path)
    return participants

def get_participant_nr(participant):
    """
    Get the participant number based on their name.

    Args:
        participant (str): Participant name.

    Returns:
        int: Participant number.
    """
    return get_participants().index(participant) + 1

