import argparse
import os

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--code', type=str)

    args = parser.parse_args()

    items = pd.read_parquet(os.path.join(args.vocab, 'items.parquet'))
