import pandas as pd
import numpy as np


def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    data = ' '.join(lines)

    return data