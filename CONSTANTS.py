import os.path
from pathlib import Path


class CONSTANTS:
    ROOT = str(Path.home())
    if "Users" in ROOT:
        HOME = f"/Users/mturja/Downloads/HCP_PTN1200"
        CODEDIR = os.path.join(ROOT, "PycharmProjects/DeepGraphKoopmanOperator")
        GraphDMDDIR = os.path.join(ROOT, "GraphDMD")
        DATADIR = f"/Users/mturja/"
    elif "longleaf" in ROOT:
        HOME = f"/work/users/m/t/mturja/HCP_PTN1200"
        CODEDIR = os.path.join("/work/users/m/t/mturja", "DeepGraphKoopmanOperator")
        GraphDMDDIR = os.path.join("/work/users/m/t/mturja", "GraphDMD")
        DATADIR = f"/work/users/m/t/mturja/"
    else:
        HOME = f"/home/mturja/HCP_PTN1200"
        CODEDIR = os.path.join("/home/mturja", "DeepGraphKoopmanOperator")
        GraphDMDDIR = os.path.join("/home/mturja", "GraphDMD")
        DATADIR = f"/home/mturja/"
