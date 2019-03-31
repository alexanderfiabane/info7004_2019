#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def main(dataset):
    data = pd.read_csv(dataset)
    print("Spliting data... ")
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=5)
    with open('train.txt', 'w') as FOUT:
        np.savetxt(FOUT, train_data, fmt='%s')
    with open('test.txt', 'w') as FOUT:
        np.savetxt(FOUT, test_data, fmt='%s')
if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Use: split_dataset.py <data>")
    main(sys.argv[1])
