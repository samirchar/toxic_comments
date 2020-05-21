import os
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.abspath("../.."))
# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import modin.pandas as pd
from src.code_snippets.utils.data_handler import read_pickle

if __name__ == "__main__":

    from src.features.build_features import textProcessor
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name",type=str)
    parser.add_argument("--file_source_path",type=str,help="Available sources are: 'raw','interim'")
    parser.add_argument("--mode",type=str,help="Available modes are transform or fit_transform")
    parser.add_argument("--max_len", type=int, default=115)
    parser.add_argument("--get_splits", type=bool, default=False)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--nrows", type=int, default=-1)
    parser.add_argument("--embedding_path",type=str,default = "../../data/raw/pretrained_embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt")
    parser.add_argument("--processed_data_path",type=str,default = "../../data/processed")
    parser.add_argument("--raw_data_path",type=str,default = "../../data/raw")
    parser.add_argument("--interim_data_path",type=str,default = "../../data/interim")

    aux_cols = [
        "num_characters",
        "num_words",
        "clean",
        "uppercase_pct",
        "lowercase_pct",
        "exclamation_pct",
        "punctuation_pct",
        "nouns_pct",
        "adjetives_pct",
        "pronouns_pct",
        "verbs_pct",
    ]

    args = parser.parse_args()
    
    if args.file_source_path == 'raw':
        file_source = args.raw_data_path
    elif args.file_source_path == 'interim':
        file_source = args.interim_data_path
    else:
        print("This file source is not available")

    if args.nrows == -1:
        args.nrows = None
    if args.get_splits:
        df = pd.read_csv(os.path.join(args.raw_data_path,"train.csv"))
        df_train, df_val = train_test_split(
            df, test_size=args.test_size, random_state=100
        )
        df_train.to_csv(os.path.join(args.interim_data_path,"train.csv"), index=False)
        df_val.to_csv(os.path.join(args.interim_data_path,"val.csv"), index=False)
    
    tp = textProcessor(
        data_dir=os.path.join(file_source,args.file_name),
        embedding_dir=args.embedding_path,
        nrows=args.nrows,
    )

    tp.read()
    tp.clean("comment_text", "comment_cleaned")
    tp.feature_engineer("comment_text")
    tp.df["y"] = (~tp.df["clean"].astype(bool)).astype(int).values


    print(f"max comment length is {args.max_len}")

    if args.mode == "fit_transform":
        tp.pre_process(
            "comment_cleaned", max_len=args.max_len, aux_cols=aux_cols, target_cols=["y"],filter_by_max_length = True
        )

    elif args.mode == "transform":

        StandardScaler_train = read_pickle(os.path.join(args.processed_data_path,"StandardScaler_train.pickle"))

        tp.pre_process(
            "comment_cleaned", max_len=args.max_len, aux_cols=aux_cols, target_cols=["y"],standard_scaler=StandardScaler_train,filter_by_max_length = True
        )
    else:
        print(f"Mode {args.mode} is not supported. Available modes are: 'transform', 'fit_transform' ")

    #Kill corenlp server
    tp.server.kill_server()

    # Save
    tp.save('.'.join(args.file_name.split('.')[0:-1]))