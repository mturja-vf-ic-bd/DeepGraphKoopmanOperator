import random
random.seed(10)


with open(f"run_mega.sh", "w") as f:
    for i in range(5):
        seed = random.randint(1, 1000)
        for dmd_rank in [2048]:
            f.writelines(f"python3 -m trainers.SparseKNF --seed={seed} -m 50 --dmd_rank={dmd_rank} --lr=0.0003 --batch_size=32 --stride=1 --input_dim=8 --input_length=128 --decay=1.0 --num_steps=32 --num_sins 10 --num_poly 2 --num_exp 2 --latent_dim=64 --hidden_dim=1024 --num_layers=3 --train_output_length=32 --test_output_length=40 --use_revin --jumps=128 --min_epochs=50 --mode=train --sbs_channel 128 --dataset megatrawl --num_feats 50 --sbs_nlayer 2 --add_global_operator &> SpKNF_mega_rank{dmd_rank}.txt")
            f.write("\n")