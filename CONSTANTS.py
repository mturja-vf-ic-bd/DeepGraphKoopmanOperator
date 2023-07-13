import os.path
from pathlib import Path


class CONSTANTS:
    ROOT = str(Path.home())
    if "Users" in ROOT:
        HOME = f"/Users/mturja/Downloads/HCP_PTN1200"
        CODEDIR = os.path.join(ROOT, "PycharmProjects/KVAE")
        GraphDMDDIR = os.path.join(ROOT, "GraphDMD")
    elif "longleaf" in ROOT:
        HOME = f"/work/users/m/t/mturja/HCP_PTN1200"
        CODEDIR = os.path.join("/work/users/m/t/mturja", "KVAE")
        GraphDMDDIR = os.path.join("/work/users/m/t/mturja", "GraphDMD")
    else:
        HOME = f"/home/mturja/HCP_PTN1200"
        CODEDIR = os.path.join("/home/mturja", "KVAE")
        GraphDMDDIR = os.path.join("/home/mturja", "GraphDMD")
