import os
import matplotlib.pyplot as plt
import matplotlib

def create_folders(folder_list, new_dir):
    paths = []
    for folder in folder_list:
        path = f"{new_dir}/{folder}"
        make_dir(path)
        paths.append(path)
    return paths

def make_dir(path):
    if not os.path.exists(path):
            os.makedirs(path)


def save_plot(fig, folder, file_name, file_extention, par=None):
    if par:
        path = f"saved-data/{par}/{folder}"
    else:
        path = f"{folder}"
        
    make_dir(path)
    fig.savefig(f"{path}/{file_name}{file_extention}")

    plt.close(fig)

def clear_figs():
    allfignums = matplotlib.pyplot.get_fignums()
    for i in allfignums:
        fig = matplotlib.pyplot.figure(i)
        fig.clear()
        matplotlib.pyplot.close( fig )