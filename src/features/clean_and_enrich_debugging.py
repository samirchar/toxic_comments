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
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
from src.features.build_features_debugging import textProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=3000)
    parser.add_argument("--get_splits", type=bool, default=False)
    parser.add_argument("--test_size", type=float, default=0.04)
    parser.add_argument("--nrows", type=int, default=-1)
    parser.add_argument("--embedding_path",type=str,default = "../../../../pretrained_embeddings/glove.twitter.27B/glove.twitter.27B.25d.txt")

    args = parser.parse_args()
    if args.nrows == -1:
        args.nrows = None
    if args.get_splits:
        df = pd.read_csv("../../data/raw/train.csv")
        df_train, df_val = train_test_split(
            df, test_size=args.test_size, random_state=100
        )
        df_train.to_csv("../../data/interim/train.csv", index=False)
        df_val.to_csv("../../data/interim/val.csv", index=False)

    tp = textProcessor(
        data_dir="../../data/interim/train.csv",
        # TODO: Change embedding size
        embedding_dir=args.embedding_path,
        nrows=args.nrows,
    )

    tp.read()
    tp.clean("comment_text", "comment_cleaned")
    tp.feature_engineer("comment_text")
    tp.df["y"] = (~tp.df["clean"].astype(bool)).astype(int).values

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

    print(f"max comment length is {args.max_len}")
    (X_indices_train, X_aux_train, y_train) = tp.pre_process(
        "comment_cleaned", max_len=args.max_len, aux_cols=aux_cols, target_cols=["y"]
    )

    ss = StandardScaler()
    ss.fit(X_aux_train)
    X_aux_train = ss.transform(X_aux_train)

    tp = textProcessor(
        data_dir="../../data/interim/val.csv",
        embedding_dir="../../../../pretrained_embeddings/glove.twitter.27B/glove.twitter.27B.25d.txt",
        nrows=args.nrows,
    )

    tp.read()
    tp.clean("comment_text", "comment_cleaned")
    tp.feature_engineer("comment_text")
    tp.df["y"] = (~tp.df["clean"].astype(bool)).astype(int).values

    (X_indices_val, X_aux_val, y_val) = tp.pre_process(
        "comment_cleaned", max_len=args.max_len, aux_cols=aux_cols, target_cols=["y"]
    )
    X_aux_val = ss.transform(X_aux_val)

    # Train
    np.save("../../data/processed/X_indices_train.npy", X_indices_train)
    np.save("../../data/processed/X_aux_train.npy", X_aux_train)
    np.save("../../data/processed/y_train.npy", y_train)

    # Validation
    np.save("../../data/processed/X_indices_val.npy", X_indices_val)
    np.save("../../data/processed/X_aux_val.npy", X_aux_val)
    np.save("../../data/processed/y_val.npy", y_val)
