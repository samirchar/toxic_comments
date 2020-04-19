import os
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.abspath("../.."))
# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)
if __name__ == "__main__":
    from src.features.build_features import textProcessor
    import modin.pandas as pd

    tp = textProcessor(
        data_dir="../../data/interim/train.csv",
        # TODO: Change embedding size
        embedding_dir="../../data/raw/pretrained_embeddings/glove.twitter.27B/glove.twitter.27B.25d.txt",
        nrows=None,
    )

    print('\n\n========== 1 =========\n\n')

    tp.read()

    print('\n\n========== 2 =========\n\n')

    tp.clean("comment_text", "comment_cleaned")

    print('\n\n========== 3 =========\n\n')

    tp.feature_engineer("comment_text")

    print('\n\n========== 4 =========\n\n')
    
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
