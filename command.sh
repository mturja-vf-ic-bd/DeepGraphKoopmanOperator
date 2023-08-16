python3 -m trainers.MegaTrawl -g 1 -d 1 -m 1000 --batch_size 200 --n_real_modes 3 --n_complex_modes 11 --lr 0.001 --num_pred_steps 5 --exp_name dgko_pred_new=5 --encoder_num_layers 7 --decoder_num_layers 7 --transformer_num_layers 7 &
python3 -m trainers.ReconOnly -g 1 -d 1 -m 1000 --batch_size 200 --lr 0.001 --exp_name recon_only --encoder_num_layers 7 --decoder_num_layers 7
