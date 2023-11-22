# Train on Megatrawl
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --latent_dim 48 --n_real_modes 5 --n_complex_modes 8 --hidden_dim 512 --num_layers 4 --use_revin --use_instancenorm --add_global_operator --add_control --jumps 128 &> varKNFexp.txt
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 64 --stride 8 --input_dim 8 --input_length 512 --decay 0.95 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 16 --use_revin --jumps 128 --mode test

# Train on HCP Task fMRI
python3 -m trainers.varKNF -m 100 --lr 0.0001 --mode train --batch_size 128 --stride 4 --num_feats 58 --input_dim 4 --dataset "hcp_task" --input_length 60 --output_length 12 --decay 0.95 --num_steps 12 --latent_dim 16 --n_real_modes 1 --n_complex_modes 6 --hidden_dim 1024 --num_layers 3 --use_revin --jumps 72 --num_sins 5 --num_poly 2 --num_exp 2 &>> hcp_task_4_24_12.txt

# Train with input_dim=16, input_length=64, 128, 256, 384, 512
python3 -m trainers.varKNF -m 50 --lr 0.0003 --mode train --batch_size 12 --stride 16 --input_dim 16 --input_length 128 --decay 0.95 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 32 --use_revin --jumps 64 --min_epochs 10 &> input_length128_input_dim16.txt
python3 -m trainers.varKNF -m 50 --lr 0.0003 --mode train --batch_size 24 --stride 16 --input_dim 16 --input_length 256 --decay 0.95 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 32 --use_revin --jumps 64 --min_epochs 10 &> input_length256_input_dim16.txt
python3 -m trainers.varKNF -m 50 --lr 0.0003 --mode train --batch_size 24 --stride 16 --input_dim 16 --input_length 384 --decay 0.95 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 32 --use_revin --jumps 64 --min_epochs 10 &> input_length384_input_dim16.txt
python3 -m trainers.varKNF -m 50 --lr 0.0003 --mode train --batch_size 32 --stride 16 --input_dim 16 --input_length 512 --decay 0.95 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 32 --use_revin --jumps 64 --min_epochs 10 &> input_length512_input_dim16.txt

python3 -m trainers.varKNF --seed=901 -m 50 --lr 0.0003 --batch_size 32 --stride 16 --input_dim 16 --input_length 128 --decay 0.9 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 32 --use_revin --jumps 64 --min_epochs 10 --add_global_operator --mode train &> DKO_dim16_rm5_inp128_global.txt
python3 -m trainers.varKNF --seed=901 -m 50 --lr 0.0003 --mode train --batch_size 32 --stride 16 --input_dim 16 --input_length 256 --decay 0.98 --num_steps 32 --latent_dim 64 --n_real_modes 7 --n_complex_modes 12 --hidden_dim 1024 --num_layers 3 --output_length 32 --use_revin --jumps 64 --min_epochs 10 &> DKO_dim16_inp256.txt
python3 -m trainers.varKNF --seed=901 -m 50 --lr 0.0003 --mode train --batch_size 80 --stride 32 --input_dim 32 --input_length 384 --decay 0.98 --num_steps 32 --latent_dim 64 --n_real_modes 5 --n_complex_modes 10 --hidden_dim 1024 --num_layers 3 --output_length 32 --use_revin --jumps 64 --min_epochs 10 --mode test &> DKO_dim32_inp384_rm20.txt
python3 -m trainers.varKNF --seed=901 -m 50 --lr 0.001 --mode train --batch_size 32 --stride 4 --input_dim 8 --input_length 128 --decay 0.98 --num_steps 32 --latent_dim 64 --n_real_modes 1 --n_complex_modes 32 --hidden_dim 1024 --num_layers 3 --train_output_length=32 --test_output_length=32 --use_revin --jumps=128 --min_epochs=30 --add_global_operator &> DKO_dim8_stride4_inp128_rm1_cm32_global.txt

python3 -m trainers.varKNF --seed=901 -m 50 --lr 0.0005 --mode train --batch_size 64 --stride 32 --input_dim 32 --input_length 256 --decay 0.98 --num_steps 32 --latent_dim 64 --n_real_modes=5 --n_complex_modes=10 --hidden_dim=1024 --num_layers=3 --train_output_length=32 --test_output_length=32 --use_revin --jumps=64 --min_epochs=50 --add_global_operator &> DKO_Megatrawl_dim32_inp384_rm5.txt

# M4 dataset
python3 -m trainers.varKNF --seed=901 --data_freq="Weekly" --dataset="M4" --train_output_length=10 --test_output_length=13 --input_dim=5 --stride=5 --input_length=45 --n_real_modes 10 --n_complex_modes 10 --hidden_dim=256 --num_layers=5 --latent_dim=64 --num_sins=4 --num_poly=4 --num_exp=2 --num_feats=1 --lr=0.0003 --batch_size=1024 --jumps=3 --decay_rate=0.85 &>> DKO_Megatrawl_dim32_inp384_rm5.txt
python3 -m trainers.varKNF --seed=901 --data_freq="Daily" --dataset="M4" --output_length=6 --input_dim=3 --input_length=18 --hidden_dim=128 --num_layers=4 --transformer_dim=128 --control_hidden_dim=64 --latent_dim=8 --lr=0.005 --batch_size=256 --jumps=5 --batch_size=1408 --decay_rate=0.85 --num_steps=6 --num_sins=2 &

python3 -m trainers.varKNF --seed=901 --data_freq="Weekly" --dataset="M4" --train_output_length=10 --test_output_length=13 --input_dim=5 --stride=5 --input_length=90 --n_real_modes 10 --n_complex_modes 10 --hidden_dim=256 --num_layers=5 --latent_dim=64 --num_sins=4 --num_poly=4 --num_exp=2 --num_feats=1 --lr=0.0003 --batch_size=1024 --jumps=3 --decay_rate=0.85 &>> DKO_M4_dim5_inp90_rm10.txt
python3 -m trainers.varKNF --seed=901 --data_freq="Daily" --dataset="M4" --output_length=6 --input_dim=3 --input_length=18 --hidden_dim=128 --num_layers=4 --transformer_dim=128 --control_hidden_dim=64 --latent_dim=8 --lr=0.005 --batch_size=256 --jumps=5 --batch_size=1408 --decay_rate=0.85 --num_steps=6 --num_sins=2 &


# DDMD
python3 -m trainers.DeepDMD --seed=902 -m 100 --dmd_rank=64 --lr=0.0003 --batch_size=512 --stride=4 --input_dim=8 --input_length=128 --decay=1.0 --num_steps=32 --latent_dim=128 --hidden_dim=1024 --num_layers=3 --train_output_length=32 --test_output_length=32 --use_revin --jumps=128 --min_epochs=50 --mode=train &> DDMD_dim8_stride4_inp128_seed902_rank64Ortho.txt
