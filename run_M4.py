import random
random.seed(100)
mode = "test"

with open(f"run_M4.sh", "w") as f:
    for i in range(5):
        seed = random.randint(1, 1000)
        for dmd_rank in [16]:
            b = 2048
            f.writelines(f"python3 -m trainers.DeepDMD --seed={seed} -m 100 --dmd_rank {dmd_rank} --lr=0.0003 --batch_size={b} --stride=5 --input_dim=5 --input_length=45 --decay=0.95 --num_steps=5 --num_sins 3 --num_poly 2 --num_exp 2 --latent_dim=64 --hidden_dim=256 --num_layers=5 --train_output_length=10 --test_output_length=13 --use_revin --jumps=3 --min_epochs=50 --mode={mode} --dataset M4 --num_feats 1 --add_global_operator")
            f.write("\n")