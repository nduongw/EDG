CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 0 1 \
       --test_envs 2 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 0 2 \
       --test_envs 1 3 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 0 3 \
       --test_envs 1 2 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 1 2 \
       --test_envs 0 3 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 1 3 \
       --test_envs 0 2 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 2 3 \
       --test_envs 0 1 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 1 2 3 \
       --test_envs 0 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 0 2 3 \
       --test_envs 1 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 0 1 3 \
       --test_envs 2 \
       --output_dir ./domainbed/output/

CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.train\
       --data_dir ./domainbed/DATA/ \
       --algorithm ERM \
       --dataset PACS \
       --train_envs 0 1 2 \
       --test_envs 3 \
       --output_dir ./domainbed/output/
       