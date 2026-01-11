import torch

from utils.cleaner import clean
from utils.menus import show_menu
from NNs.optimiserForNN import optimise_nn
from NNs.optimiserForCNN import optimise_cnn
from NNs.mainRunner import make_run

process_options = ["Clean datasets",  #0-1
                   "Optimise hyperparameters",  #1-2
                   "Train Modul",  #2-3
                   "Exit"]

def main():
    print(f"PyTorch version: {torch.__version__}")  # Ensure it's a CUDA-compatible version
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version {torch.version.cuda}")
    else:
        print(f"Device: CPU")
    print(f"")
    while True:
        process = show_menu("Select process by entering its number: ", process_options)
        if process == process_options[0]:
            clean()
        elif process == process_options[1]:
            type = int(input())
            if type == 1:
                optimise_nn()
            elif type == 2:
                optimise_cnn()
        elif process == process_options[2]:
            make_run()
        else:
            break

# Using the special variable
# __name__
if __name__=="__main__":
    main()