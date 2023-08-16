import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from dataloaders.MegaTrawl import MegaTrawlDataModule
from models.KNF import KNF_Embedder, compute_prediction_loss
from utils.folder_manager import create_version_dir


class trainerSparseEdgeKoopman(pl.LightningModule):
    def __init__(self,
                 input_dim,
                 input_length,
                 u_window_size,
                 encoder_hidden_dim=256,
                 decoder_hidden_dim=256,
                 encoder_num_layers=5,
                 decoder_num_layers=5,
                 n_real_dmd_modes=1,
                 n_complex_dmd_mode_pairs=15,
                 transformer_dim=256,
                 transformer_num_layers=5,
                 num_heads=1,
                 latent_dim=24,
                 num_sins=2,
                 num_poly=2,
                 num_exp=0,
                 num_feats=50,
                 lr=1e-4,
                 num_steps=1,
                 loss_weights: List[int] = None):
        super(trainerSparseEdgeKoopman, self).__init__()

        self.input_dim = input_dim
        self.input_length = input_length
        self.u_window_size = u_window_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.n_real_dmd_modes = n_real_dmd_modes
        self.n_complex_dmd_mode_pairs = n_complex_dmd_mode_pairs
        self.transformer_dim = transformer_dim
        self.transformer_num_layers = transformer_num_layers
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.num_sins = num_sins
        self.num_poly = num_poly
        self.num_exp = num_exp
        self.num_feats = num_feats
        self.learning_rate = lr
        self.num_steps = num_steps
        self.model = KNF_Embedder(
            input_dim=input_dim,
            input_length=input_length,
            u_window_size=u_window_size,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_num_layers=encoder_num_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            n_real_dmd_modes=n_real_dmd_modes,
            n_complex_dmd_mode_pairs=n_complex_dmd_mode_pairs,
            transformer_dim=transformer_dim,
            transformer_num_layers=transformer_num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
            num_sins=num_sins,
            num_poly=num_poly,
            num_exp=num_exp,
            num_feats=num_feats
        )
        self.loss_weights = [1, 1, 1, 1, 1, 1] if loss_weights is None \
                                            else loss_weights
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

    def _common_step(self, batch, batch_idx, category="train"):
        x, s = batch
        V_cur, V_future, X_cur, X_future, \
        X_recon_cur, X_recon_future, \
            embeddings_cur, embeddings_future, \
                dmd_modes, omega_cur, omega_future, \
                    pred_list_cur, pred_list_future, \
                        latent_pred_list_cur, latent_pred_list_future = \
                            self.model(x, max_k=self.num_steps)
        mse_loss = nn.MSELoss(reduction='mean')
        loss_recon = mse_loss(X_recon_cur, X_cur) + mse_loss(X_recon_future, X_future)
        loss_latent_pred_cur = compute_prediction_loss(
            embeddings_cur.flatten(start_dim=2), latent_pred_list_cur)
        loss_latent_pred_future = compute_prediction_loss(
            embeddings_future.flatten(start_dim=2), latent_pred_list_future)
        loss_pred_cur = compute_prediction_loss(x[:, :, :self.u_window_size].transpose(1, 2), pred_list_cur)
        loss_pred_future = compute_prediction_loss(x[:, :, self.u_window_size:].transpose(1, 2), pred_list_future)
        loss_ortho = torch.matmul(dmd_modes.transpose(1, 2), dmd_modes)
        loss_ortho = torch.mean(torch.triu(loss_ortho, diagonal=1) ** 2) * 2
        if self.current_epoch > -1:
            loss = self.loss_weights[0] * loss_recon + \
                   self.loss_weights[1] * loss_latent_pred_cur + \
                   self.loss_weights[2] * loss_latent_pred_future + \
                   self.loss_weights[3] * loss_pred_cur + \
                   self.loss_weights[4] * loss_pred_future + \
                   self.loss_weights[5] * loss_ortho
        else:
            loss = self.loss_weights[0] * loss_recon
        losses = {
            "loss/recon": loss_recon,
            "loss/pred_latent_cur": loss_latent_pred_cur,
            "loss/pred_latent_future": loss_latent_pred_future,
            "loss/pred_cur": loss_pred_cur,
            "loss/pred_future": loss_pred_future,
            "loss/ortho": loss_ortho,
            "loss/total": loss}
        self.add_log(losses, category)
        progress_bar = {f"{category}_total": loss.item(),
                        f"{category}_recon": loss_recon.item(),
                        f"{category}_pred_cur": loss_latent_pred_future.item(),
                        f"{category}_ortho": loss_ortho.item(),
                        f"{category}_pred_future": loss_latent_pred_future.item()}
        self.losses[category].append(loss)
        return {'loss': loss, 'progress_bar': progress_bar}

    def predict_step(self, batch, batch_idx):
        x, s = batch
        # x = x[:, :, 0:(x.shape[2] // self.sw_size) * self.input_dim]
        # x = x.unfold(-1, self.sw_size, self.sw_size).permute(0, 2, 1, 3).flatten(end_dim=1)
        V_cur, V_future, X_cur, X_future, \
        X_recon_cur, X_recon_future, \
        embeddings_cur, embeddings_future, \
        dmd_modes, omega_cur, omega_future, \
        pred_list_cur, pred_list_future, \
        latent_pred_list_cur, latent_pred_list_future = \
            self.model(x, max_k=self.num_steps)
        X_recon_cur = X_recon_cur.reshape(X_recon_cur.shape[0], X_recon_cur.shape[1], -1,
                                          self.num_feats).reshape(X_recon_cur.shape[0],
                                                                  -1, self.num_feats).transpose(1, 2)
        plt.figure(figsize=(50, 50))
        for i in range(10):
            plt.subplot(10, 1, i + 1)
            plt.plot(x[1, i, 0:800].detach().cpu().numpy(), c='b')
            plt.plot(X_recon_cur[2, i].detach().cpu().numpy(), c='r', linestyle="--")
        plt.show()

        return pred_list_cur, pred_list_future


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
    parser.add_argument("--write_dir", type=str, default="log/DeepGraphKoopman/MegaTrawl")
    parser.add_argument("--weight", type=float, nargs="+")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--input_length", type=int, default=1200)
    parser.add_argument("--u_window_size", type=int, default=800)
    parser.add_argument("--num_pred_steps", type=int, default=5)
    parser.add_argument("--n_real_modes", type=int, default=1)
    parser.add_argument("--n_complex_modes", type=int, default=7)
    parser.add_argument("--encoder_num_layers", type=int, default=5)
    parser.add_argument("--decoder_num_layers", type=int, default=5)
    parser.add_argument("--transformer_num_layers", type=int, default=5)

    arguments = parser.parse_args()
    print("Hyper-parameters:")
    for k, v in vars(arguments).items():
        print("{} -> {}".format(k, v))

    _HOME = os.path.expanduser('~')
    data_loader = MegaTrawlDataModule(batch_size=arguments.batch_size)
    accelerator = "cpu" if arguments.gpus == 0 else "gpu"
    device = arguments.device

    if arguments.mode == "train":
        # Create write dir
        write_dir = create_version_dir(
            os.path.join(arguments.write_dir, arguments.exp_name),
            prefix="run")
        arguments.write_dir = write_dir
        ckpt = ModelCheckpoint(dirpath=os.path.join(write_dir, "checkpoints"),
                               monitor="val/loss/total",
                               every_n_epochs=5,
                               save_top_k=5,
                               save_last=True,
                               auto_insert_metric_name=False,
                               filename='epoch-{epoch:02d}-recon_loss-{val/loss/recon:0.3f}-pred_loss-{'
                                        'val/loss/pred_cur:0.3f}')
        tb_logger = pl_loggers.TensorBoardLogger(write_dir, name="tb_logs")
        trainer = pl.Trainer(accelerator=accelerator,
                             devices=device,
                             max_epochs=arguments.max_epochs,
                             logger=tb_logger,
                             log_every_n_steps=5,
                             callbacks=[ckpt])
        if not arguments.from_ckpt:
            model = trainerSparseEdgeKoopman(
                input_dim=arguments.input_dim,
                input_length=arguments.input_length,
                u_window_size=arguments.u_window_size,
                loss_weights=arguments.weight,
                num_feats=arguments.num_nodes,
                num_steps=arguments.num_pred_steps,
                lr=arguments.lr,
                n_real_dmd_modes=arguments.n_real_modes,
                n_complex_dmd_mode_pairs=arguments.n_complex_modes,
                encoder_num_layers=arguments.encoder_num_layers,
                decoder_num_layers=arguments.decoder_num_layers,
                transformer_num_layers=arguments.transformer_num_layers
            )
        else:
            model = trainerSparseEdgeKoopman.load_from_checkpoint(arguments.ckpt, map_location=torch.device('cpu'))
        trainer.fit(
            model,
            train_dataloaders=data_loader.train_dataloader(),
            val_dataloaders=data_loader.val_dataloader()
        )
        best_model = trainerSparseEdgeKoopman.load_from_checkpoint(ckpt.best_model_path)
        return predict(data_loader.val_dataloader(), best_model), arguments, model
    else:
        data_loader = data_loader.val_dataloader()
        model = trainerSparseEdgeKoopman.load_from_checkpoint(arguments.ckpt, map_location=torch.device('cpu'))
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
