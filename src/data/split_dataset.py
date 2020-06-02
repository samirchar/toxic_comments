import os
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.abspath("../.."))
# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)
# import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--file_source_path", type=str, help="Available sources are: 'raw','interim'")
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--nrows", type=int, default=-1)
    parser.add_argument("--processed_data_path", type=str, default="../../data/processed")
    parser.add_argument("--raw_data_path", type=str, default="../../data/raw")
    parser.add_argument("--interim_data_path", type=str, default="../../data/interim")
    parser.add_argument("--splits",type=str,default="train val test")
    args = parser.parse_args()

    if args.file_source_path == "raw":
        file_source = args.raw_data_path
    elif args.file_source_path == "interim":
        file_source = args.interim_data_path
    else:
        print("This file source is not available")

    if args.nrows == -1:
        args.nrows = None
        
    df = pd.read_csv(os.path.join(args.raw_data_path, "dataset.csv"))
    
    if args.splits=='train val':
        df_train, df_val = train_test_split(
            df,
            test_size=args.test_size,
            stratify = (df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]==1).any(axis=1),
            random_state=123
        )

        df_train.to_csv(os.path.join(args.interim_data_path, "train.csv"), index=False)
        df_val.to_csv(os.path.join(args.interim_data_path, "val.csv"), index=False)
    
    elif args.splits == 'train val test':

        df_train, df_temp = train_test_split(df,
            test_size=args.test_size*2,
            stratify = (df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]==1).any(axis=1),
            random_state=123)

        df_val, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify = (df_temp[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]==1).any(axis=1),
            random_state=123)

        df_train.to_csv(os.path.join(args.interim_data_path, "train.csv"), index=False)
        df_val.to_csv(os.path.join(args.interim_data_path, "val.csv"), index=False)
        df_test.to_csv(os.path.join(args.interim_data_path, "test.csv"), index=False)