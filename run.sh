# Train on Megatrawl
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --latent_dim 48 --n_real_modes 5 --n_complex_modes 8 --hidden_dim 512 --num_layers 4 --use_revin --use_instancenorm --add_global_operator --add_control --jumps 128 &> varKNFexp.txt
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --stride 8 --input_dim 8 --input_length 512 --decay 0.95 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 16 --use_revin --jumps 128 --mode test

# Train on HCP Task fMRI
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 128 --stride 4 --num_feats 58 --input_dim 4 --dataset "hcp_task" --input_length 60 --output_length 12 --decay 0.95 --num_steps 12 --latent_dim 16 --n_real_modes 1 --n_complex_modes 6 --hidden_dim 1024 --num_layers 3 --use_revin --jumps 72 --num_sins 5 --num_poly 2 --num_exp 2 &>> hcp_task_4_24_12.txt

# Train with input_dim=16, input_length=64, 128, 256, 384, 512
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --stride 16 --input_dim 16 --input_length 128 --decay 0.95 --num_steps 64 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 64 --use_revin --jumps 64
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --stride 16 --input_dim 16 --input_length 256 --decay 0.95 --num_steps 64 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 64 --use_revin --jumps 64
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --stride 16 --input_dim 16 --input_length 284 --decay 0.95 --num_steps 64 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 64 --use_revin --jumps 64
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --stride 16 --input_dim 16 --input_length 512 --decay 0.95 --num_steps 64 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 64 --use_revin --jumps 64
