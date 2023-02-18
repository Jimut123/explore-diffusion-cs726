import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from collections import OrderedDict
from torch import optim, nn, utils, Tensor

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")
print("Device to be used : ",device)


class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expected and is being saved correctly in `hparams.yaml`.
        """
        
        self.time_embed = self._time_embed

        # 5,32,64,64,3 == model 1
        # self.model_1 = nn.Sequential(nn.Linear(5, 32), 
        #                            nn.ReLU(), 
        #                            nn.Linear(32, 64), 
        #                            nn.ReLU(), 
        #                            nn.Linear(64, 64), 
        #                            nn.ReLU(), 
        #                            nn.Linear(64, 3)
        #                            )
        
        # tested on this model (2) for q-1, semi complex.
        self.model = nn.Sequential(nn.Linear(5, 64), 
                                   nn.ReLU(), 
                                   nn.Linear(64, 128), 
                                   nn.ReLU(), 
                                   nn.Linear(128, 256), 
                                   nn.ReLU(), 
                                   nn.Linear(256, 64),
                                   nn.ReLU(), 
                                   nn.Linear(64, 3)
                                   )
        
        # model 3
        # self.model_3 = nn.Sequential(nn.Linear(5, 16), 
        #                            nn.ReLU(), 
        #                            nn.Linear(16, 32), 
        #                            nn.ReLU(), 
        #                            nn.Linear(32, 64), 
        #                            nn.ReLU(), 
        #                            nn.Linear(64, 32),
        #                            nn.ReLU(), 
        #                            nn.Linear(32, 16),
        #                            nn.ReLU(), 
        #                            nn.Linear(16, 3)
        #                            )

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        self.betas, self.alphas, self.alpha_bars = self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
             t = torch.LongTensor([t]).expand(x.size(0))
        t_embed = self.time_embed(t)
        input_model = torch.cat((x, t_embed), dim=1).float().to(device)
        
        return self.model(input_model)


    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        betas = torch.linspace(start = lbeta, end = ubeta, steps = self.n_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim = 0)
        return betas, alphas, alpha_bars
    

    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        """
        norm = torch.randn_like(x).to(device)
        t = t.reshape(-1).long()
        ab = self.alpha_bars[t]
        ab = ab.reshape([x.shape[0]]+(len(x.shape)-1)*[1]).to(device)
        return ab.sqrt() * x + (1 - ab).sqrt() * norm, norm

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        t_tensor = self.time_embed(t*torch.ones((x.shape[0], 1)))
        xt_app = torch.cat((x, t_tensor), dim = 1)

        beta = self.betas[t]
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]

        mod_res = self.model(xt_app)
        term1 = beta * mod_res / (1 - alpha_bar).sqrt()
        term1 = (x - term1) / alpha.sqrt()

        norm = beta.sqrt() * torch.randn_like(term1)
        term2 = norm if t > 0 else 0
        return term1 + term2


    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.
        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        X_T = batch
        X_T = X_T.to(torch.float32).to(device)
        t = np.random.randint(0, self.n_steps, (batch.shape[0], ))
        t_tensor = torch.Tensor(t).reshape((-1, 1))
        X_noise,noise=self.q_sample(X_T,t)
        X_T_pred = self.forward(X_T.to(device),torch.tensor(t).to(device))
        loss = nn.functional.mse_loss(X_T_pred, noise)
        return loss
        

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        xt = torch.tensor(np.random.randn(n_samples, self.n_dim)).float()
        ls = [xt.cpu().detach().numpy()]
        if progress:
            for t in tqdm(range(self.n_steps), ascii = True):
                xt = self.p_sample(xt, self.n_steps - t - 1)
                if return_intermediate:
                    ls.append(xt.cpu().detach().numpy())
        else:
             for t in range(self.n_steps):
                xt = self.p_sample(xt, self.n_steps - t - 1)
                if return_intermediate:
                    ls.append(xt.cpu().detach().numpy())
        return xt.cpu().detach() if not return_intermediate else xt.cpu().detach(), torch.tensor(np.array(ls))

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer