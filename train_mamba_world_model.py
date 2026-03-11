import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import datetime

# ==========================================
# 1. 数据集定义 (保持原样)
# ==========================================
class DroneTrajectoryDataset(Dataset):
    def __init__(self, data_dir, seq_len=30):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=False)
        
        images = data['img']
        amodal_segs = data['amodal_seg'] 
        occ_ratios = data['occlusion_ratio'] 
        actions = data['action']
        
        total_frames = images.shape[0]
        if self.seq_len < total_frames:
            start_idx = np.random.randint(0, total_frames - self.seq_len + 1)
        else:
            start_idx = 0
            
        end_idx = start_idx + self.seq_len
        
        img_slice = images[start_idx:end_idx]
        img_tensor = torch.from_numpy(img_slice).float().permute(0, 3, 1, 2) / 255.0
        
        seg_slice = amodal_segs[start_idx:end_idx]
        target_mask = (seg_slice == 1).astype(np.float32) 
        mask_tensor = torch.from_numpy(target_mask).unsqueeze(1) 
        
        action_tensor = torch.from_numpy(actions[start_idx:end_idx]).float()
        occ_tensor = torch.from_numpy(occ_ratios[start_idx:end_idx]).float()
        
        return img_tensor, mask_tensor, action_tensor, occ_tensor


# ==========================================
# 2. 核心创新：纯 PyTorch 版选择性状态空间引擎 (Mamba Core)
# ==========================================
class SelectiveSSMCell(nn.Module):
    """
    原生实现的 Mamba 核心单元。替换传统 GRU，原生支持连续物理时间建模与动态输入过滤。
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 状态转移矩阵 A (代表物理世界的连续运动规律，通常初始化为负数以保证系统稳定)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A)) 

        # 核心突破：B, C 和 Delta 矩阵不再是固定的，而是根据当前输入 x_t 动态生成的！
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)
        self.proj_Delta = nn.Linear(d_model, d_model)

        # 跨层跳跃连接
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x_t, ssm_state_prev):
        """
        x_t: 当前帧的特征输入 (Batch, d_model)
        ssm_state_prev: 上一帧的高维信念状态矩阵 (Batch, d_model, d_state)
        """
        A = -torch.exp(self.A_log) # 恢复为负数 (d_model, d_state)

        # 1. 动态生成选择性参数 (这就是 "Selective" 的奥义)
        B_t = self.proj_B(x_t) # (Batch, d_state)
        C_t = self.proj_C(x_t) # (Batch, d_state)
        # Delta_t 相当于时间步长/门控。遇到遮挡柱子时，网络会自发将其降至接近 0
        Delta_t = nn.functional.softplus(self.proj_Delta(x_t)) # (Batch, d_model)

        # 2. 连续方程离散化 (Zero-order hold)
        Delta_A = Delta_t.unsqueeze(-1) * A.unsqueeze(0)
        Delta_B = Delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
        A_bar = torch.exp(Delta_A) # 严格的数学转换

        # 3. 物理状态更新
        # 如果 Delta_t 接近 0，A_bar 接近 1，Delta_B 接近 0。
        # 此时 ssm_state_t ≈ ssm_state_prev，完美实现“忽略当前帧噪声，依靠惯性推演”！
        ssm_state_t = A_bar * ssm_state_prev + Delta_B * x_t.unsqueeze(-1)

        # 4. 输出当前帧的综合特征 (Belief)
        y_t = torch.sum(ssm_state_t * C_t.unsqueeze(1), dim=-1) + self.D * x_t

        return y_t, ssm_state_t


# ==========================================
# 3. 算法架构：概率化选择性世界模型 (SSM-VAE 架构)
# ==========================================
class ProbabilisticSelectiveWorldModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=6, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_dim = latent_dim // 2 
        self.ego_dim = latent_dim // 2    
        self.hidden_dim = hidden_dim      
        
        # --- A. 视觉特征提取器 ---
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512), nn.ReLU()
        )
        
        # --- B. 概率分布网络 ---
        self.posterior_net = nn.Linear(512 + hidden_dim, latent_dim * 2)
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        
        # --- C. 动力学信念追踪 (重构为 Selective SSM) ---
        # 预处理投影：将隐变量和动作融合，对齐到 SSM 的维度
        self.ssm_input_proj = nn.Sequential(
            nn.Linear(self.latent_dim + action_dim, self.hidden_dim),
            nn.SiLU() # Mamba 偏爱的激活函数
        )
        # 彻底抛弃低端的 GRUCell
        self.dynamics_ssm = SelectiveSSMCell(d_model=self.hidden_dim, d_state=16)
        
        # --- D. Amodal 掩码解码器 ---
        self.mask_decoder_fc = nn.Linear(self.target_dim, 128 * 16 * 16)
        self.mask_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_step(self, img_t, action_prev, z_prev, ssm_state_prev):
        """单步前向传播：利用 SSM 引擎进行推演"""
        # 1. 提取综合输入并更新 SSM 物理状态
        ssm_in = self.ssm_input_proj(torch.cat([z_prev, action_prev], dim=-1))
        
        # belief_t 等价于原先的 h_t，但底层是强大的微分方程支撑
        belief_t, ssm_state_t = self.dynamics_ssm(ssm_in, ssm_state_prev)
        
        # 2. 计算先验分布 (基于 SSM 提炼的信念)
        prior_stats = self.prior_net(belief_t)
        mu_prior, logvar_prior = torch.chunk(prior_stats, 2, dim=-1)
        
        # 3. 提取画面并计算后验
        cnn_feat = self.cnn_encoder(img_t)
        post_input = torch.cat([cnn_feat, belief_t], dim=-1)
        post_stats = self.posterior_net(post_input)
        mu_post, logvar_post = torch.chunk(post_stats, 2, dim=-1)
        
        # 4. 采样与解耦重建
        z_t = self.reparameterize(mu_post, logvar_post)
        z_target_t = z_t[:, :self.target_dim]
        pred_mask_logits = self.decode_mask(z_target_t)
        
        return pred_mask_logits, z_t, ssm_state_t, (mu_prior, logvar_prior), (mu_post, logvar_post)

    def decode_mask(self, z_target):
        x = self.mask_decoder_fc(z_target)
        x = x.view(-1, 128, 16, 16)
        return self.mask_decoder_conv(x)


# ==========================================
# 4. 训练逻辑与损失函数设计
# ==========================================
def kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
    return 0.5 * kl.sum(dim=-1) 

def train_mamba_world_model():
    # 实验超参数
    DATA_DIR = "../isaac_project/STWM_Dataset_V11_Amodal" # 换成你的绝对路径
    BATCH_SIZE = 8
    SEQ_LEN = 30
    EPOCHS = 50
    LR = 3e-4
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    LAMBDA_RECON = 1.0
    LAMBDA_KL = 0.1 
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"mamba_world_model_{timestamp}")
    writer = SummaryWriter(log_dir)
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"🚀 启动 SSM 架构训练 | 设备: {DEVICE} | 日志目录: {log_dir}")
    
    full_dataset = DroneTrajectoryDataset(DATA_DIR, seq_len=SEQ_LEN)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)
    
    model = ProbabilisticSelectiveWorldModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    mask_criterion = nn.BCEWithLogitsLoss(reduction='none')
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss, total_kl_loss, total_loss = 0, 0, 0
        
        for imgs, masks, actions, occ_ratios in train_loader:
            imgs, masks, actions = imgs.to(DEVICE), masks.to(DEVICE), actions.to(DEVICE)
            occ_ratios = occ_ratios.to(DEVICE)
            batch_size, seq_len, _, _, _ = imgs.shape
            
            optimizer.zero_grad()
            batch_recon_loss, batch_kl_loss = 0.0, 0.0
            
            # 【核心修改】初始化 SSM 的三维物理状态矩阵 (Batch, d_model, d_state)
            ssm_state_prev = torch.zeros(batch_size, model.hidden_dim, 16).to(DEVICE)
            z_prev = torch.zeros(batch_size, model.latent_dim).to(DEVICE)
            action_prev = torch.zeros(batch_size, actions.shape[-1]).to(DEVICE)
            
            for t in range(seq_len):
                pred_mask_logits, z_t, ssm_state_t, prior_stats, post_stats = model.forward_step(
                    imgs[:, t], action_prev, z_prev, ssm_state_prev
                )
                
                mu_prior, logvar_prior = prior_stats
                mu_post, logvar_post = post_stats
                
                bce_loss_raw = mask_criterion(pred_mask_logits, masks[:, t])
                occ_penalty_weight = 1.0 - 0.6 * occ_ratios[:, t].view(batch_size, 1, 1, 1)
                weighted_bce_loss = (bce_loss_raw * occ_penalty_weight).mean()
                batch_recon_loss += weighted_bce_loss
                
                kl_loss = kl_divergence_gaussian(mu_post, logvar_post, mu_prior, logvar_prior).mean()
                batch_kl_loss += kl_loss
                
                # 更新状态
                z_prev, ssm_state_prev, action_prev = z_t, ssm_state_t, actions[:, t]
            
            batch_recon_loss /= seq_len
            batch_kl_loss /= seq_len
            loss = LAMBDA_RECON * batch_recon_loss + LAMBDA_KL * batch_kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += batch_recon_loss.item()
            total_kl_loss += batch_kl_loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, masks, actions, occ_ratios in val_loader:
                imgs, masks, actions = imgs.to(DEVICE), masks.to(DEVICE), actions.to(DEVICE)
                occ_ratios = occ_ratios.to(DEVICE)
                batch_size, seq_len, _, _, _ = imgs.shape
                
                v_loss_recon, v_loss_kl = 0.0, 0.0
                
                # 同步更新验证集的 SSM 状态矩阵
                ssm_state_prev = torch.zeros(batch_size, model.hidden_dim, 16).to(DEVICE)
                z_prev = torch.zeros(batch_size, model.latent_dim).to(DEVICE)
                action_prev = torch.zeros(batch_size, actions.shape[-1]).to(DEVICE)
                
                for t in range(seq_len):
                    pred_mask_logits, z_t, ssm_state_t, prior_stats, post_stats = model.forward_step(
                        imgs[:, t], action_prev, z_prev, ssm_state_prev
                    )
                    
                    mu_prior, logvar_prior = prior_stats
                    mu_post, logvar_post = post_stats
                    
                    bce_loss_raw = mask_criterion(pred_mask_logits, masks[:, t])
                    occ_penalty_weight = 1.0 - 0.6 * occ_ratios[:, t].view(batch_size, 1, 1, 1)
                    v_loss_recon += (bce_loss_raw * occ_penalty_weight).mean()
                    v_loss_kl += kl_divergence_gaussian(mu_post, logvar_post, mu_prior, logvar_prior).mean()
                    
                    z_prev, ssm_state_prev, action_prev = z_t, ssm_state_t, actions[:, t]
                    
                total_val_loss += (LAMBDA_RECON * (v_loss_recon / seq_len) + LAMBDA_KL * (v_loss_kl / seq_len)).item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
        writer.add_scalar("Loss/Train_Recon_Amodal", total_recon_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/Train_KL_Dynamics", total_kl_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/Validation_Total", avg_val_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "checkpoints/best_mamba_world_model.pth"
            torch.save(model.state_dict(), save_path)

    writer.close()
    print("🎉 SSM 训练管线跑通测试完毕！")

if __name__ == "__main__":
    train_mamba_world_model()