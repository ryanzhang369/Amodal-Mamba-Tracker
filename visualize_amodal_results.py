import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from train_mamba_world_model import ProbabilisticSelectiveWorldModel

def visualize_qualitative_results():
    DATA_DIR = "../isaac_project/STWM_Dataset_V11_Amodal"
    CHECKPOINT_PATH = "checkpoints/best_mamba_world_model.pth"
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print(f"正在加载预训练 Mamba 模型...")
    model = ProbabilisticSelectiveWorldModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    test_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.npz")))
    if len(test_files) == 0:
        raise FileNotFoundError("找不到测试数据，请检查路径！")
    
    print("正在扫描数据集，寻找拥有最高遮挡率的 Episode...")
    best_file = None
    best_peak_idx = 0
    max_occ_value = 0.0
    
    for f in test_files:
        data = np.load(f, allow_pickle=False)
        occ = data['occlusion_ratio']
        if np.max(occ) > max_occ_value:
            max_occ_value = np.max(occ)
            best_file = f
            best_peak_idx = np.argmax(occ)
    
    print(f"选定文件: {os.path.basename(best_file)}")
    print(f"最高遮挡率: {max_occ_value:.2f} (发生在第 {best_peak_idx} 帧)")

    test_data = np.load(best_file, allow_pickle=False)
    images_np = test_data['img']
    actions_np = test_data['action']
    
    SEQ_LEN = min(300, len(images_np)) 
    
    img_tensor = torch.from_numpy(images_np[:SEQ_LEN]).float().permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE) / 255.0
    action_tensor = torch.from_numpy(actions_np[:SEQ_LEN]).float().unsqueeze(0).to(DEVICE)
    
    predicted_probs = []
    batch_size = 1
    ssm_state_prev = torch.zeros(batch_size, model.hidden_dim, 16).to(DEVICE)
    z_prev = torch.zeros(batch_size, model.latent_dim).to(DEVICE)
    action_prev = torch.zeros(batch_size, action_tensor.shape[-1]).to(DEVICE)
    
    with torch.no_grad():
        for t in range(SEQ_LEN):
            pred_mask_logits, z_t, ssm_state_t, _, _ = model.forward_step(
                img_tensor[:, t], action_prev, z_prev, ssm_state_prev
            )
            prob_map = torch.sigmoid(pred_mask_logits).squeeze().cpu().numpy()
            predicted_probs.append(prob_map)
            z_prev, ssm_state_prev, action_prev = z_t, ssm_state_t, action_tensor[:, t]

    print("正在生成定性结果对比图...")
    num_display = 8
    
    start_frame = max(0, best_peak_idx - 35)
    end_frame = min(SEQ_LEN - 1, best_peak_idx + 40)
    frame_indices = np.linspace(start_frame, end_frame, num_display, dtype=int)
    
    fig, axes = plt.subplots(2, num_display, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    cmap = plt.get_cmap('inferno')
    
    for i, f_idx in enumerate(frame_indices):
        ax_img = axes[0, i]
        ax_img.imshow(images_np[f_idx])
        ax_img.axis('off')
        if i == 0:
            ax_img.set_title("Input: Drone RGB View", fontsize=14, loc='left', pad=10)
        
        ax_heat = axes[1, i]
        heat_prob = predicted_probs[f_idx]
        gray_img = np.mean(images_np[f_idx], axis=-1) / 255.0
        ax_heat.imshow(gray_img, cmap='gray', alpha=0.5)
        
        masked_heat = np.ma.masked_where(heat_prob < 0.1, heat_prob)
        im = ax_heat.imshow(masked_heat, cmap=cmap, alpha=0.8, vmin=0, vmax=1.0)
        ax_heat.axis('off')
        if i == 0:
            ax_heat.set_title("Ours: Mamba Amodal Heatmap", fontsize=14, loc='left', pad=10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.3]) 
    fig.colorbar(im, cax=cbar_ax, label="Belief Confidence")

    save_path = "amodal_qualitative_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"图已保存至: {save_path}")

if __name__ == "__main__":
    visualize_qualitative_results()
