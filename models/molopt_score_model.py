import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from utils.vina_rules import calc_vina

def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return gumbel_noise + logits


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.loss_exp_weight = config.loss_exp_weight

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        
        self.use_classifier_guide = config.use_classifier_guide

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # classifier guidance weight
        self.pos_classifier_grad_weight = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )
        self.expert_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        batch_size = batch_protein.max().item() + 1
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif len(init_ligand_v.shape) == 2:
            pass
        else:
            raise ValueError
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)
        
        if 'pred_exp_from_all' in self.config and self.config.pred_exp_from_all:
            atom_affinity = self.expert_pred(final_h).squeeze(-1)
            final_exp_pred = scatter_mean(atom_affinity, batch_all)
        else:
            atom_affinity = self.expert_pred(final_ligand_h).squeeze(-1)
            final_exp_pred = scatter_mean(atom_affinity, batch_ligand)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'atom_affinity': atom_affinity,
            'final_exp_pred': final_exp_pred,
            'batch_all': batch_all,
            'mask_ligand': mask_ligand
        }
        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, protein_pos, protein_v, affinity, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        
        loss_exp = F.mse_loss(preds['final_exp_pred'], affinity)
        
        if self.use_classifier_guide:
            loss = loss_pos + loss_v * self.loss_v_weight + loss_exp * self.loss_exp_weight
        else:
            loss = loss_pos + loss_v * self.loss_v_weight
            
        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_exp': preds['final_exp_pred'],
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['final_ligand_h']
        }

    def calc_atom_dis(
            self, protein_pos, protein_v, affinity, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        time_step_arr = torch.arange(0,1001,20).to(protein_pos.device)
        time_step_arr[-1] = 999
        lig_pro_dis_all = []
        for time_step in tqdm(time_step_arr):
            time_step = torch.tensor(time_step.tolist()).repeat(num_graphs).to(protein_pos.device)
            a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

            # 2. perturb pos and v
            a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
            pos_noise = torch.zeros_like(ligand_pos)
            pos_noise.normal_()
            # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
            ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
            # Vt = a * V0 + (1-a) / K
            lig_pro_dis = []
            for batch_idx in range(num_graphs):
                
                pro_coords = protein_pos[batch_protein==batch_idx]
                pro_ident = torch.tensor([0]).repeat(pro_coords.shape[0],1)
            #     print(loader_data)
                mol_coords = ligand_pos_perturbed[batch_ligand==batch_idx]
                mol_ident = torch.tensor([1]).repeat(mol_coords.shape[0],1)
                all_coords = torch.cat((mol_coords,pro_coords),dim=0)
                all_ident = torch.cat((mol_ident, pro_ident),dim=0)
                all_ident_rep = all_ident.T.repeat(all_coords.shape[0],1)
            #         all_coords_ident = torch.cat((all_coords, all_ident), dim=1)
                dist = torch.sum((all_coords[:,None,:] - all_coords[None,:,:])**2,dim=-1).sqrt()
                _, ind = torch.sort(dist, 1)
                # num_lig_atom = []
                num_lig_atom = all_ident[ind][:,:32].sum(dim=1)
                # for dis, indice in zip(all_ident_rep, ind):
                #     num_lig_atom.append(dis[indice][:32].sum(dim=0,keepdim=True))
                # num_lig_atom = torch.stack(num_lig_atom,dim=0)
                num_lig_atom_ident = torch.cat((all_ident, num_lig_atom),dim=1)
                lig_pro_dis.append(num_lig_atom_ident.cpu())
            
            lig_pro_dis = torch.cat(lig_pro_dis,dim=0)
            lig_pro_dis_all.append(lig_pro_dis)
        # lig_pro_dis_all = torch.cat(lig_pro_dis_all,dim=0)
        
        torch.save(lig_pro_dis_all, 'knn32_atom_type_num_across_1000step.pt')
    # def classifier_gradient(self, input, batch_all, t):
    #     with torch.enable_grad():
    #         x_in = input.detach().requires_grad_(True)
    #         final_exp_pred = scatter_mean(self.expert_pred(x_in).squeeze(-1), batch_all)
    #         grad = torch.autograd.grad(final_exp_pred, x_in,grad_outputs=torch.ones_like(final_exp_pred))[0]
    #         # w is same across dimension 0 at t step
    #         w = self.classifier_grad_weight[t].to(input.device)[0]
    #         # print(x_in.shape, final_exp_pred.shape, w.shape, grad.shape,batch_all.shape,batch_all)
    #         return grad / torch.sqrt(w**2+1)
    
    def pv_joint_guide(self, ligand_v_index, ligand_pos, protein_v, protein_pos, batch_protein, batch_ligand):
        with torch.enable_grad():
            ligand_v = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
            
            init_h_protein = self.protein_atom_emb(protein_v)
            init_ligand_h = self.ligand_atom_emb(ligand_v)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            h_protein = torch.cat([init_h_protein, torch.zeros(len(init_h_protein), 1).to(init_h_protein)], -1)
            ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(init_ligand_h)], -1)

            h_all, pos_all, batch_all, mask_ligand = compose_context(
                h_protein=h_protein,
                h_ligand=ligand_h,
                pos_protein=protein_pos,
                pos_ligand=ligand_pos,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
            )

            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all)
            final_pos, final_h = outputs['x'], outputs['h']
            final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
            
            
            if 'pred_exp_from_all' in self.config and self.config.pred_exp_from_all:
                atom_affinity = self.expert_pred(final_h).squeeze(-1)
                pred_affinity = scatter_mean(atom_affinity, batch_all)
            else:
                atom_affinity = self.expert_pred(final_ligand_h).squeeze(-1)
                pred_affinity = scatter_mean(atom_affinity, batch_ligand)
            # pred_affinity = scatter_mean(self.expert_pred(final_h).squeeze(-1), batch_all)

            
            pred_affinity_log = pred_affinity.log()
            
            type_grad = torch.autograd.grad(pred_affinity, ligand_v,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            pos_grad = torch.autograd.grad(pred_affinity_log, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
        
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'atom_affinity': atom_affinity,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'final_exp_pred': pred_affinity,
            'batch_all': batch_all,
            'mask_ligand': mask_ligand,
        }
        return preds, type_grad, pos_grad
                
            
    # def classifier_gradient(self, input, batch_all, t):
    #     with torch.enable_grad():
    #         x_in = input.detach().requires_grad_(True)
    #         final_exp_pred = scatter_mean(self.expert_pred(x_in).squeeze(-1), batch_all)
    #         grad = torch.autograd.grad(final_exp_pred, x_in,grad_outputs=torch.ones_like(final_exp_pred))[0]
    #         return grad
        
    def vina_classifier_gradient(self, logits_ligand_v_recon, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t):
        
        with torch.enable_grad():
            x_in = logits_ligand_v_recon.detach().requires_grad_(True)
            ligand_pos_in = ligand_pos.detach().requires_grad_(True)
            
            vina_score, vina_score_each = calc_vina(F.gumbel_softmax(x_in,hard=True,tau=0.5), ligand_pos_in, protein_v, protein_pos, batch_ligand, batch_protein)
            grad1 = torch.autograd.grad(vina_score, x_in,grad_outputs=torch.ones_like(vina_score), create_graph=True)[0]
            grad2 = torch.autograd.grad(vina_score, ligand_pos_in,grad_outputs=torch.ones_like(vina_score), create_graph=True)[0]
            return grad1, grad2, vina_score_each
        
    def value_net_classifier_gradient(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t, value_model):
        value_model.eval()
        with torch.enable_grad():
            
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob,hard=True,tau=0.5)
            preds = value_model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                time_step=t
            )
            pred_affinity = preds['final_exp_pred']
            
            grad1 = torch.autograd.grad(pred_affinity, ligand_v_next_prob,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            grad2 = torch.autograd.grad(pred_affinity, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            return grad1, grad2, pred_affinity
        
    def value_net_classifier_gradient_rep(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t, value_model):
        value_model.eval()
        with torch.enable_grad():
            
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob,hard=True,tau=0.5)
            preds = value_model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                time_step=t
            )
            pred_affinity = preds['final_exp_pred']
            
            grad1 = torch.autograd.grad(pred_affinity, ligand_v_next_prob,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            grad2 = torch.autograd.grad(pred_affinity, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            w = self.pos_classifier_grad_weight[t].to(ligand_v_next_prob.device)[0]
            return grad1 / torch.sqrt(w**2+1), grad2 / torch.sqrt(w**2+1), pred_affinity

    def value_net_classifier_gradient_rep2(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t, value_model):
        value_model.eval()
        with torch.enable_grad():
            
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob,hard=True,tau=0.5)
            preds = value_model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                time_step=t
            )
            pred_affinity = preds['final_exp_pred']
            
            grad1 = torch.autograd.grad(pred_affinity, ligand_v_next_prob,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            grad2 = torch.autograd.grad(pred_affinity, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            w2 = self.pos_classifier_grad_weight[t].to(ligand_v_next_prob.device)[0]
            w1 = self.log_alphas_v[t].exp().to(ligand_v_next_prob.device)[0]
            return grad1 * w1, grad2 / torch.sqrt(w2**2+1), pred_affinity
        
    # @torch.no_grad()
    def sample_diffusion(self, guide_mode, type_grad_weight, pos_grad_weight, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,value_model=None,
                         num_steps=None, center_pos_mode=None):
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj, exp_traj, exp_atom_traj = [], [], [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random':
                preds, type_grad, pos_grad = self.pv_joint_guide(ligand_v, ligand_pos, protein_v, protein_pos, batch_protein, batch_ligand)
            elif guide_mode == 'target_diff':
                preds = self(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,

                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )
                pred = value_model(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,

                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )
            
            else:
                preds = self(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,

                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )
                
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            # pos posterior
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            
            # type posterior
            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
            
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random':
                exp_pred = preds['final_exp_pred']
                
                pos_model_mean = pos_model_mean + pos_grad_weight*(0.5 * pos_log_variance).exp()*pos_grad
                log_ligand_v = log_ligand_v + type_grad_weight*type_grad
                
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                
            elif guide_mode == 'vina':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.vina_classifier_gradient(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t)
                ligand_v_next_prob = ligand_v_next_prob - grads1 * type_grad_weight
                ligand_pos = ligand_pos - grads2 * pos_grad_weight
                
            elif guide_mode == 'valuenet':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.value_net_classifier_gradient(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t,value_model)
                ligand_v_next_prob = ligand_v_next_prob + grads1 * type_grad_weight
                ligand_pos = ligand_pos + grads2 * pos_grad_weight
            
            elif guide_mode == 'target_diff':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                exp_pred = pred['final_exp_pred']
                preds['atom_affinity'] = pred['atom_affinity']
                
            elif guide_mode == 'valuenet_rep' or guide_mode == 'valuenet_rep1':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.value_net_classifier_gradient_rep(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t,value_model)
                ligand_v_next_prob = ligand_v_next_prob + grads1 * type_grad_weight
                ligand_pos = ligand_pos + grads2 * pos_grad_weight
            
            elif guide_mode == 'valuenet_rep2':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.value_net_classifier_gradient_rep2(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t,value_model)
                ligand_v_next_prob = ligand_v_next_prob + grads1 * type_grad_weight
                ligand_pos = ligand_pos + grads2 * pos_grad_weight
                   
            elif guide_mode == 'wo':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                exp_pred = None
            else:
                raise NotImplementedError
            
            ligand_v_next = ligand_v_next_prob.argmax(dim=-1)
            ligand_v = ligand_v_next
                
            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(ligand_v_next_prob.clone().cpu())   
                
            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
            
            if exp_pred is not None:
                exp_traj.append(exp_pred.clone().cpu())
                exp_atom_traj.append(preds['atom_affinity'].clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'exp': exp_traj[-1] if len(exp_traj) else [],
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_traj': exp_traj,
            'exp_atom_traj': exp_atom_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

# %%
