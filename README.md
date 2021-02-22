#  APIrecX
This our submission repo in ACL 2021


## Install
Use the following command to install the environment required by APIrecX 

```
conda env create -f pytorch-gpu.yaml
```

## Usage
#### pretrain quick start
first clone this repository:
```
git clone https://github.com/anonymous98514/APIrecX.git
```
Then enter the pre-training folder 
```
cd pretrain
python -u main.py --train_corpus {raw data path} --vocab_file {bpe tokenizer vocabulary} --batch_size 32 --pretrained_sp_model {bpe tokenizer model} --local_rank 1 --n_layers 6 --lr 1.5e-4 --n_attn_heads 8 --epochs 15 --max_seq_len 512 --hidden 256 --ffn_hidden 512 --model_output_path {output path} --pretrain
```

#### fine_tune
Use the following command to quickly start APIrecX fine tuning 
```
cd finetune
python -u train.py --epoch 15 --lr 2e-3 --weight_decay 1e-8  --batch_size 8 --sample {sample ratio} --max_seq_len 512 --device_index 0 --k 10 --is_save False --boundary {beam size}
```
#### baseline

```
```

## Contributing

PRs accepted.

## License

MIT © Richard McRichface
