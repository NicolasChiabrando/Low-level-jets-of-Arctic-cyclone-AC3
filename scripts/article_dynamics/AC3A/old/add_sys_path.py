""" for the issues with source root directory in Jupyter or terminal, to import my libraries from the folder Libs."""

import os
import sys

sys.path.insert(1, os.path.join(os.getcwd(),
                                ".."))  # to get up of one level. If we want to access also a file two level up, need "../.."
sys.path.insert(2, os.path.join(os.getcwd(),
                                "../.."))
sys.path.insert(3, os.path.join(os.getcwd(),
                                "../../.."))
sys.path.insert(4, os.path.join(os.getcwd(),
                                "../../../.."))
sys.path.insert(5, os.path.join(os.getcwd(),
                                "../../../../.."))

print(sys.path)