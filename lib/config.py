import os
import sys
from easydict import EasyDict

CONF = EasyDict()
# scalar
CONF.SCALAR = EasyDict()
CONF.SCALAR.OBJ_PC_SAMPLE = 1000    # or 1024 in pointnet config
CONF.SCALAR.REL_PC_SAMPLE = 3000

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/shreyasm/pigraph/"
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data/")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# 3RScan data
CONF.PATH.R3Scan = os.path.join(CONF.PATH.DATA, "scans")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")