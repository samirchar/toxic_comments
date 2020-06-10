python clean_and_enrich.py --file_name train.csv --file_source_path interim --mode fit_transform
python clean_and_enrich.py --file_name test.csv --file_source_path interim --mode transform
python clean_and_enrich.py --file_name val.csv --file_source_path interim --mode transform