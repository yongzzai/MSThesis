export TORCH_USE_CUDA_DSA=0
export TORCH_CUDNN_V8_API_ENABLED=1



python main.py --embed_dim 16 --hidden_dim 64 \
 --num_enc_layers 4 --num_dec_layers 2 \
 --enc_dropout 0.3 --dec_dropout 0.3 \
 --batch_size 64 --epochs 18 \
 --lr 0.0003 --seed 42

python main.py --embed_dim 16 --hidden_dim 64 \
 --num_enc_layers 4 --num_dec_layers 2 \
 --enc_dropout 0.3 --dec_dropout 0.3 \
 --batch_size 32 --epochs 18 \
 --lr 0.0003 --seed 42

python main.py --embed_dim 16 --hidden_dim 64 \
 --num_enc_layers 4 --num_dec_layers 2 \
 --enc_dropout 0.3 --dec_dropout 0.3 \
 --batch_size 16 --epochs 18 \
 --lr 0.0003 --seed 42

python main.py --embed_dim 16 --hidden_dim 64 \
 --num_enc_layers 4 --num_dec_layers 2 \
 --enc_dropout 0.3 --dec_dropout 0.3 \
 --batch_size 8 --epochs 18 \
 --lr 0.0003 --seed 42

wait

echo "Done."