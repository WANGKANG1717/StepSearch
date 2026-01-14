data_root=/data1/wk/search-r1/StepSearch/data/musi
dataset_name=musi

python search_r1/search/retrieval_rerank_server.py \
    --data_root $data_root \
    --port 8000 \
    --dataset_name $dataset_name

    # --faiss_gpu \
