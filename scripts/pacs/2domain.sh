CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_env 1 2 \
       --test_env 3 4 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_env 1 3 \
       --test_env 2 4 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_env 1 4 \
       --test_env 2 4 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_env 2 3 \
       --test_env 1 4 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_env 2 4 \
       --test_env 1 3 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_env 3 4 \
       --test_env 1 2 \
       --output_dir ./domainbed/output/