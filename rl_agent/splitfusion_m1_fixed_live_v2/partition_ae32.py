import sys
from .contracts import AE32_UINT8_A54 as C
from .partition_common import apply

if __name__ == "__main__":
    apply(C, sys.argv[1], sys.argv[2])

