import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning import loggers as pl_loggers

from dataloaders.MegaTrawl import MegaTrawlDataModule
from models.DeepKoopman import KoopmanAutoencoder
from utils.folder_manager import create_version_dir
from utils.correlation import pearson


class trainerSparseEdgeKoopman(pl.LightningModule):
    def __init__(self,
                 encoder_width_list: List[int] = None,
                 decoder_width_list: List[int] = None,
                 edge_proj_width_list: List[int] = None,
                 edge_func_enc_width: List[int] = None,
                 edge_func_dec_width: List[int] = None,
                 edge_func_proj_width: List[int] = None,
                 num_nodes=50,
                 look_back_window=10,
                 lr=1e-4,
                 loss_weights=None):
        super(trainerSparseEdgeKoopman, self).__init__()

        if encoder_width_list is None:
            encoder_width_list = [16, 128, 128, 128, 32]
        if decoder_width_list is None:
            decoder_width_list = encoder_width_list[::-1]
        if edge_proj_width_list is None:
            edge_proj_width_list = [1, 16, 16, 1]
        if edge_func_enc_width is None:
            edge_func_enc_width = [256, 128, 128, 128, 32]
        if edge_func_dec_width is None:
            edge_func_dec_width = edge_func_enc_width[::-1]
        if edge_func_proj_width is None:
            edge_func_proj_width = [1, 16, 16, 1]

        self.model = KoopmanAutoencoder(
            num_nodes=num_nodes,
            num_real_modes=3,
            num_complex_modes=4,
            delta_t=0.72,
            encoder_width_list=encoder_width_list,
            decoder_width_list=decoder_width_list,
            edge_proj_width_list=edge_proj_width_list,
            edge_func_enc_width=edge_func_enc_width,
            edge_func_dec_width=edge_func_dec_width,
            edge_func_proj_width=edge_func_proj_width,
            activation=nn.ReLU()
        )
        self.learning_rate = lr
        self.sw_size = edge_func_enc_width[0]
        self.look_back_window = look_back_window
        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1, 1, 1]
        self.losses = {"train": [], "val": []}
        self.save_hyperparameters()

    def forward(self, input):
        return self.model(input)

    def add_log(self, losses, category="train"):
        for k, v in losses.items():
            self.log(category + "/" + k, v)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        loss = sum(self.losses["train"]) / len(self.losses["train"])
        self.losses["train"] = []
        print(f"Training ... total loss:{loss:0.3f}")
        return {'avg_epoch_loss': loss}

    def on_validation_epoch_end(self):
        loss = sum(self.losses["val"]) / len(self.losses["val"])
        self.losses["val"] = []
        print(f"Validation ... total loss:{loss:0.3f}")
        return {'avg_epoch_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=0.001)

    def get_attn(self):
        return self.model.K

    def _common_step(self, batch, batch_idx, category="train"):
        x, s = batch
        x = x[:, :, 0:(x.shape[2] // self.sw_size) * self.sw_size]
        x = x.unfold(-1, self.sw_size, self.sw_size).permute(0, 2, 1, 3).flatten(end_dim=1)
        y, g, g_next_list, U, omega, x_recon = self.model(x.float(), max_k=self.look_back_window)
        loss_recon = nn.MSELoss(reduction='mean')(x, x_recon)
        loss_pred = 0
        for i, g_next in enumerate(g_next_list):
            loss_pred += nn.MSELoss(reduction='mean')(g[:, :, i+1:], g_next[:, :, :-(i+1)])
        loss_pred /= len(g_next_list)
        loss_ortho = torch.matmul(U.transpose(1, 2), U)
        loss_ortho = torch.mean(torch.triu(loss_ortho, diagonal=1) ** 2) * 2
        loss = self.loss_weights[0] * loss_recon + \
               self.loss_weights[1] * loss_pred + \
               self.loss_weights[2] * loss_ortho
        losses = {
            "loss/recon": loss_recon,
            "loss/pred": loss_pred,
            "loss/ortho": loss_ortho,
            "loss/total": loss}
        self.add_log(losses, category)
        progress_bar = {f"{category}_total": loss,
                        f"{category}_recon": loss_recon,
                        f"{category}_pred": loss_pred,
                        f"{category}_ortho": loss_ortho}
        self.losses[category].append(loss)
        return {'loss': loss, 'progress_bar': progress_bar, 'log': progress_bar}

    def predict_step(self, batch, batch_idx):
        x, s = batch
        x = x[:, :, 0:(x.shape[2] // self.sw_size) * self.sw_size]
        x = x.unfold(-1, self.sw_size, self.sw_size).permute(0, 2, 1, 3).flatten(end_dim=1)
        y, g, g_next_list, U, omega, x_recon = self.model(x.float())
        return y, g, g_next_list, U, omega, x_recon


def cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", nargs="?", type=int, default=0)
    parser.add_argument("-m", "--max_epochs", nargs="?", type=int, default=100)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--exp_name", type=str, default="fmri",
                        help="Name you experiment")
    parser.add_argument("--write_dir", type=str, default="log/DeepGraphKoopman/MegaTrawl")
    parser.add_argument("--weight", type=float, nargs="+")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=50)

    arguments = parser.parse_args()
    print("Hyper-parameters:")
    for k, v in vars(arguments).items():
        print("{} -> {}".format(k, v))

    _HOME = os.path.expanduser('~')
    data_loader = MegaTrawlDataModule(batch_size=arguments.batch_size)
    accelerator = "cpu" if arguments.gpus == 0 else "gpu"
    device = 1

    if arguments.mode == "train":
        # Create write dir
        write_dir = create_version_dir(
            os.path.join(arguments.write_dir, arguments.exp_name),
            prefix="run")
        arguments.write_dir = write_dir
        ckpt = ModelCheckpoint(dirpath=os.path.join(write_dir, "checkpoints"),
                               monitor="val/loss/total",
                               every_n_epochs=5,
                               save_top_k=1,
                               save_last=True,
                               auto_insert_metric_name=False,
                               filename='epoch-{epoch:02d}-recon_loss-{val/loss/recon:0.3f}-lkis_loss-{'
                                        'val/loss/pred:0.3f}')
        tb_logger = pl_loggers.TensorBoardLogger(write_dir, name="tb_logs")
        trainer = pl.Trainer(accelerator=accelerator,
                             devices=device,
                             max_epochs=arguments.max_epochs,
                             logger=tb_logger,
                             log_every_n_steps=5,
                             callbacks=[ckpt])
        if not arguments.from_ckpt:
            model = trainerSparseEdgeKoopman(
                loss_weights=arguments.weight,
                num_nodes=arguments.num_nodes,
                lr=arguments.lr
            )
        else:
            model = trainerSparseEdgeKoopman.load_from_checkpoint(arguments.ckpt)
        trainer.fit(
            model,
            train_dataloaders=data_loader.train_dataloader(),
            val_dataloaders=data_loader.val_dataloader()
        )
        best_model = trainerSparseEdgeKoopman.load_from_checkpoint(ckpt.best_model_path)
        return predict(data_loader.val_dataloader(), best_model), arguments, model
    else:
        data_loader = data_loader.val_dataloader()
        model = trainerSparseEdgeKoopman.load_from_checkpoint(arguments.ckpt)
        torch.save(model.model, "megatrawl_model_bn.ckpt")
        return predict(data_loader, model), arguments, model


def predict(data_loader, model):
    trainer = pl.Trainer()
    output = trainer.predict(model, data_loader)
    return output


def plot_set_of_multivarite_ts(T):
    def plot_multivariate_ts(ts):
        for i in range(len(ts)):
            # if max(ts[i]) > 5:
            #     plt.ylim(-5, 5)
            plt.plot(ts[i])

    plt.figure(figsize=(10, 5))
    i = 1
    for k, ts in T.items():
        plt.subplot(len(T), 1, i)
        plt.title(k)
        plot_multivariate_ts(ts)
        i = i + 1
    plt.show()


def plot_histogram(Vh):
    fig = plt.figure(figsize=(20, 20))
    N = 10
    for i in range(len(Vh)):
        ax = fig.add_subplot(N, 4, i + 1)
        ax.hist(Vh[i], bins=10)
    plt.tight_layout()
    plt.show()


def get_transient_modes(lam, z, g):
    phi = np.dot(z.conj().T, g)
    ind = np.argmin(np.abs(lam))
    return np.abs(phi[ind])


def write_output(output, n):
    G = output["G_u"]
    dummy = torch.ones(G.shape[0], G.shape[1], n, n)
    res = torch.zeros(G.shape[0], G.shape[1], n, n)
    res[torch.triu(dummy) == 1] = G.flatten()
    D = torch.diag_embed(torch.diagonal(res, dim1=-1, dim2=-2), dim1=-1, dim2=-2)
    res = res + res.transpose(-1, -2) - D
    scipy.io.savemat(
        "../loaders/fmri.mat", {"input": output["input"].detach().cpu().numpy(),
                     "fmri_mat": res.permute(0, 2, 3, 1).detach().cpu().numpy(),
                     "labels": output["labels"].detach().cpu().numpy()})


if __name__ == '__main__':
    start_time = time.time()
    # A = torch.from_numpy(load_corr_mat("../A_train_w_64_single_hcp.mat")["A"])
    # G = torch.from_numpy(load_corr_mat("../G_train_w_64_single_hcp.mat")["valid_chunks"])
    # plot_matrix(A[3], G[3])
    output, arguments, model = cli_main()
    # ind = 0
    # b_ind = 0
    # inp = output[ind]["input"]
    # Z = output[ind]["z_gauss"]
    # A = corr_coeff(inp)
    # G = corr_coeff(Z)
    # # plot_loss_against_window_size(A, G)
    #
    # fig = plt.figure(figsize=(30, 15))
    # w = 128
    #
    # for j in range(15):
    #     ax = fig.add_subplot(5, 3, j + 1)
    #     A_lkis = compute_adj_lkis_sparse_edge(A[j:j+1], w=w, stride=1, parcel=50)[2:-2]
    #     G_lkis = compute_adj_lkis_sparse_edge(G[j:j+1], w=w, stride=1, parcel=50)[2:-2]
    #     ax.set_ylim(0, min(max(A_lkis), 0.01))
    #     ax.plot(A_lkis, label="Original lkis loss")
    #     ax.plot(G_lkis, label="Latent lkis loss")
    #     ax.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # n = 100
    # t = 50
    # # res = model.model.predict_n_step(G[t, :], K, n=n)
    # # fig = plt.figure(figsize=(120, 5))
    # # size = 20
    # # # G[G < 0] = 0
    # # # A[A < 0] = 0
    # # for i in range(80):
    # #     ax = fig.add_subplot(2, 40, i+1)
    # #     # ax.set_yticks(list(np.arange(-.5, 0.1, 0.01)))
    # #     # ax.set_yticklabels(list(np.arange(-1, 1, 0.1)), fontsize=size)
    # #     # ax.set_ylim(-0.05, 0.13)
    # #     # ax.set_xticks(np.arange(0, 101, 10))
    # #     # ax.set_xticklabels(np.around(np.arange(0, 101, 10), decimals=2), fontsize=size)
    # #
    # #     # ax.plot(Z[t: t + n + 1, i], c="r", label="GT")
    # #     if i > 40:
    # #         ax.imshow(G[b_ind][i - 40], vmin=0, vmax=1)
    # #     else:
    # #         ax.imshow(A[b_ind][i], vmin=0, vmax=1)
    # #     # ax.plot(res[i], c="b", label="prediction")
    # # plt.tight_layout()
    # # plt.show()
    #
    # # K = np.linalg.lstsq(Z[1:].detach().cpu().numpy(), Z[:-1].detach().cpu().numpy())[0]
    # # # K = np.linalg.lstsq(G[1:].detach().cpu().numpy(), G[:-1].detach().cpu().numpy())[0]
    # # K = torch.Tensor(K)
    # # e, v = torch.linalg.eig(K)
    # # min_ind = torch.argmin(torch.abs(e))
    # # g_K = torch.complex(Z, torch.zeros_like(Z)) @ v
    # # g_K = g_K[:, min_ind]
    # # plt.plot(torch.abs(g_K))
    # # plt.show()
    #
    # # plot_histogram(Z.permute(1, 0)[0:10])
    # plot_set_of_multivarite_ts(
    #     {"Input": output[ind]["input"][b_ind, :, :, 0].permute(1, 0),
    #      "Reconstructed": output[ind]["recon"][b_ind, :, :, 0].permute(1, 0)
    #      # "Z": Z.permute(1, 0)
    #      })
    # # write_output(output[ind], 34)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")
