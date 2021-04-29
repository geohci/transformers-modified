python get_data.py \
    --languages en_XX,cs_CZ,fr_XX,it_IT,fi_FI,de_DE,nl_XX,ro_RO,ru_RU,ja_XX \
    --num_train 100000 \
    --num_val 10000 \
    --num_test 10000 \
    --path /content/gdrive/MyDrive/wikidata_folders/ \
    --data_dir /content/data_dir/ \
    --embd_path /content/gdrive/MyDrive/embeddings/deepwalk_wikidata.pickle \
    --graph_path /content/gdrive/MyDrive/embeddings/wikidata_graph_only_wikipedia.txt \
    "$@"