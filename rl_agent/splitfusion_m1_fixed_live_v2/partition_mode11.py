import sys
from .contracts import MODE11_UINT4_Q6000 as C
from .partition_common import apply

if __name__ == "__main__":
    apply(C, sys.argv[1], sys.argv[2])

