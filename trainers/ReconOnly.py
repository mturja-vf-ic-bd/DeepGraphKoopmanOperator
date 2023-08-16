import os
import time

import torch
import torch.nn as nn

from dataloaders.MegaTrawl import MegaTrawlDataModule
from models.ReconOnly import KNF_Embedder_Recon
from utils.folder_manager import create_version_dir

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank: int, world_size: int):
    """
    Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12375"
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size)
    torch.cuda.set_device(rank)


def run_training(
        rank,
        world_size,
        input_dim,
        input_length,
        encoder_hidden_dim,
        encoder_num_layers,
        decoder_hidden_dim,
        decoder_num_layers,
        latent_dim, num_sins,
        num_poly, num_exp,
        num_feats, save_every,
        batch_size, write_dir,
        exp_name, lr, max_epochs):
    ddp_setup(rank, world_size)
    _HOME = os.path.expanduser('~')
    data_loader = MegaTrawlDataModule(batch_size=batch_size)
    train_data = data_loader.train_dataloader()
    model = KNF_Embedder_Recon(
        input_dim=input_dim,
        input_length=input_length,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_layers=encoder_num_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=decoder_num_layers,
        latent_dim=latent_dim,
        num_sins=num_sins,
        num_poly=num_poly,
        num_exp=num_exp,
        num_feats=num_feats
    ).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = DDP(model, device_ids=[rank])
    if rank == 0:
        write_dir = create_version_dir(
            os.path.join(write_dir, exp_name),
            prefix="run")
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0
        train_data.sampler.set_epoch(epoch)
        iter = 0
        for x, s in train_data:
            optimizer.zero_grad()
            V_cur, X_cur, X_recon_cur, \
                embedding_cur = model(x)
            mse_loss = nn.MSELoss(reduction='mean')
            loss = mse_loss(X_recon_cur, X_cur)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iter += 1
        running_loss /= iter
        if rank == 0 and epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, os.path.join(write_dir, f"checkpoint_{epoch}.ckpt"))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch {epoch}: {running_loss} time = {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")


def cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", nargs="?", type=int, default=0)
    parser.add_argument("-d", "--device", type=int, default=1)
    parser.add_argument("-m", "--max_epochs", nargs="?", type=int, default=100)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--exp_name", type=str, default="knf",
                        help="Name you experiment")
    parser.add_argument("--write_dir", type=str, default="log/DeepGraphKoopman/ReconOnly")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--input_length", type=int, default=1200)
    parser.add_argument("--encoder_num_layers", type=int, default=5)
    parser.add_argument("--decoder_num_layers", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=32)

    arguments = parser.parse_args()
    world_size = torch.cuda.device_count()
    print("Hyper-parameters:")
    for k, v in vars(arguments).items():
        print("{} -> {}".format(k, v))
    mp.spawn(run_training,
             args=(world_size,
                   arguments.input_dim,
                   arguments.input_length,
                   512,
                   arguments.encoder_num_layers,
                   512,
                   arguments.decoder_num_layers,
                   arguments.latent_dim,
                   2,
                   3,
                   2,
                   arguments.num_nodes,
                   arguments.save_every,
                   arguments.batch_size,
                   arguments.write_dir,
                   arguments.exp_name,
                   arguments.lr,
                   arguments.max_epochs),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    start_time = time.time()
    cli_main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")
