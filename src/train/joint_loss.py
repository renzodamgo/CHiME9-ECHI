import torch, torch.nn.functional as F

def vad_from_tf(Y_tf, thr_db=-40.0, eps=1e-10):
    mag2 = (Y_tf[...,0,:,:]**2 + Y_tf[...,1,:,:]**2).mean(dim=-1)   # [B,K,T]
    db   = 10*torch.log10(mag2 + eps)
    return (db > thr_db)

def si_sdr(est, ref, eps=1e-8):
    ref_energy = (ref**2).sum(dim=-1, keepdim=True) + eps
    proj = (est*ref).sum(dim=-1, keepdim=True)*ref/ref_energy
    noise = est - proj
    return 10*torch.log10(((proj**2).sum(dim=-1)+eps)/((noise**2).sum(dim=-1)+eps))

class JointLoss:
    def __init__(self, spec_loss_fn, istft_fn, spk_encoder=None,
                 l_mix=0.1, l_xt=0.1, l_spk=0.1, use_sisdr=True, tau=0.07):
        self.spec_loss_fn = spec_loss_fn
        self.istft_fn = istft_fn
        self.spk_encoder = spk_encoder
        if self.spk_encoder:
            for p in self.spk_encoder.parameters(): p.requires_grad = False
        self.l_mix, self.l_xt, self.l_spk = l_mix, l_xt, l_spk
        self.use_sisdr, self.tau = use_sisdr, tau

    def __call__(self, S_hat, X_ref, Y_ref_tf, y_wav, enroll_embs, present_mask=None):
        # S_hat: [B,K,2,T,F] complex; X_ref: [B,2,T,F]; Y_ref_tf: [B,K,2,T,F]
        B, K = S_hat.shape[:2]
        device = S_hat.device
        if present_mask is None:
            present_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        # (1) separation (spec + optional SI-SDR)
        L_sep = 0.0
        for k in range(K):
            mask = present_mask[:,k]
            if mask.any():
                L_sep += self.spec_loss_fn(S_hat[:,k][mask], Y_ref_tf[:,k][mask])
        denom = max(present_mask.sum().item(), 1)
        L_sep = L_sep/denom

        if self.use_sisdr:
            s_hat_wav = self.istft_fn(S_hat)                       # [B,K,Tw]
            sisdr = si_sdr(s_hat_wav[present_mask], y_wav[present_mask]).mean()
            L_sep = L_sep - sisdr

        # (2) mixture consistency
        L_mix = F.l1_loss(S_hat.sum(dim=1), X_ref)

        # (3) cross-talk (VAD-gated)
        vad = vad_from_tf(Y_ref_tf).float()                        # [B,K,T]
        silent = (1.0 - vad)
        est_e = ((S_hat[:, :, 0]**2 + S_hat[:, :, 1]**2).mean(dim=-1))  # [B,K,T]
        pm = present_mask[:,:,None].float()
        L_xt = (silent*est_e*pm).sum() / pm.sum().clamp_min(1.0)

        # (4) speaker-agreement (InfoNCE) — optional
        if self.spk_encoder:
            s_hat_wav = self.istft_fn(S_hat)
            E_hat, E_tar = [], []
            for b in range(B):
                for k in range(K):
                    if present_mask[b,k]:
                        E_hat.append(self.spk_encoder(s_hat_wav[b,k].unsqueeze(0)))
                        E_tar.append(enroll_embs[b,k].unsqueeze(0))
            if E_hat:
                E_hat = torch.cat(E_hat,0); E_tar = torch.cat(E_tar,0)
                E_hat = F.normalize(E_hat, dim=-1); E_tar = F.normalize(E_tar, dim=-1)
                logits = (E_hat @ E_tar.t())/self.tau
                labels = torch.arange(logits.size(0), device=logits.device)
                L_spk = F.cross_entropy(logits, labels)
            else:
                L_spk = torch.tensor(0.0, device=device)
        else:
            L_spk = torch.tensor(0.0, device=device)

        total = L_sep + self.l_mix*L_mix + self.l_xt*L_xt + self.l_spk*L_spk
        return total, {"L_sep":float(L_sep), "L_mix":float(L_mix),
                       "L_xtalk":float(L_xt), "L_spk":float(L_spk), "L_total":float(total)}
