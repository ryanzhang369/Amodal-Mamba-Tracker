from omni.isaac.kit import SimulationApp

# 1. 初始化 SimulationApp
simulation_app = SimulationApp({"headless": True, "physics_dt": 1.0/120.0, "rendering_dt": 1.0/120.0})

import os
import numpy as np
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.objects import VisualCuboid 
from pxr import UsdGeom, Gf
from scipy.spatial.transform import Rotation as R
from omni.isaac.nucleus import get_assets_root_path

# ==========================================
# Hyperparameters & Config
# ==========================================
# 【改良】你可以放心大胆地设为 1000 了
NUM_EPISODES = 1000         

# 【改良】物理步进次数设为 1200 次 (10 秒)。实际保存帧数依然是 1200 / 4 = 300 帧
PHYSICS_STEPS_PER_EPISODE = 1200  
SAVE_INTERVAL = 4 # 每 4 个物理步(对应 30 FPS)保存一次数据

RESOLUTION = (256, 256)   
DT = 1.0 / 120.0          
VEHICLE_YAW_OFFSET = 90.0 
MIN_SAFE_DIST = 8.0       

base_dir = os.path.abspath(os.getcwd())
output_dir = os.path.join(base_dir, "STWM_Dataset_V11_Amodal") 
assets_dir = os.path.join(base_dir, "assets")

drone_usd = os.path.join(assets_dir, "crazyflie_final_new.usd")

DRONE_PATH = "/World/Crazyflie"
VEHICLE_BASE_PATH = "/World/Vehicle"
CAM_DRONE_PATH = "/World/Camera_Drone"

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise Exception("无法连接到 Nucleus 服务器！")
    
vehicle_usd = assets_root_path + "/Isaac/Robots/Forklift/forklift_c.usd"

# ==========================================
# 物理运动逻辑 (保持不变)
# ==========================================
class AggressiveTracker:
    def __init__(self, init_pos):
        self.curr_pos = np.array(init_pos, dtype=float)
        self.vel = np.zeros(3)
        self.omega = 16.0   
        self.zeta = 0.85    
        self.min_z = 0.2    

    def update(self, target_goal, dt):
        error = target_goal - self.curr_pos
        accel = (self.omega**2) * error - 2.0 * self.omega * self.zeta * self.vel
        max_acc = 45.0 
        acc_mag = np.linalg.norm(accel)
        if acc_mag > max_acc: accel = (accel / acc_mag) * max_acc
        self.vel += accel * dt
        next_pos = self.curr_pos + self.vel * dt
        if next_pos[2] < self.min_z:
            next_pos[2] = self.min_z
            if self.vel[2] < 0: self.vel[2] = 0
        self.curr_pos = next_pos
        return self.curr_pos, self.vel

def get_vehicle_pose(t, params):
    base_spd = params.get("base_speed", 8.0)
    dynamic_speed = base_spd * (1.0 + 0.3 * np.cos(0.4 * t)) 
    scale_x = params.get("track_length", 25.0)
    scale_y = params.get("track_width", 15.0)
    omega = 0.4 * (dynamic_speed / 6.0) 
    
    dir_flip = params.get("dir_flip", 1.0)
    x = params["center_offset"][0] + scale_x * (np.cos(omega * t) - 1.0)
    y = params["center_offset"][1] + dir_flip * scale_y * np.sin(2 * omega * t) / 2.0
    dx = -scale_x * omega * np.sin(omega * t)
    dy = dir_flip * scale_y * omega * np.cos(2 * omega * t)
    
    heading_deg = np.degrees(np.arctan2(dy, dx)) + VEHICLE_YAW_OFFSET
    return np.array([x, y, 0.1]), heading_deg

# ==========================================
# Initialize World & Replicator Lighting
# ==========================================
world = World(stage_units_in_meters=1.0)
stage = world.stage

world.scene.add_default_ground_plane() 

track_points = []
for t_test in np.linspace(0, 100, 200):
    omega_test = 0.4 
    x_test = 8.0 * (np.cos(omega_test * t_test) - 1.0) 
    y_test = 5.0 * np.sin(2 * omega_test * t_test) / 2.0 
    track_points.append(np.array([x_test, y_test]))
    track_points.append(np.array([x_test, -y_test]))
track_points = np.array(track_points)

obstacle_paths = []
obstacle_positions_2d = [] 
print("正在生成动态遮挡障碍物 (启用双重防穿模机制)...")
num_obstacles = 0

while num_obstacles < 20:  
    obs_x = np.random.uniform(-20, 5)
    obs_y = np.random.uniform(-15, 15)
    obs_pos_2d = np.array([obs_x, obs_y])
    
    distances = np.linalg.norm(track_points - obs_pos_2d, axis=1)
    min_dist = np.min(distances)
    
    if min_dist > 3.5: 
        obs_name = f"obstacle_pillar_{num_obstacles}"
        obs_path = f"/World/Obstacles/{obs_name}"
        
        VisualCuboid(
            prim_path=obs_path,
            name=obs_name,
            position=np.array([obs_x, obs_y, 2.0]),
            scale=np.array([2.0, 2.0, 4.0]),  
            color=np.array([0.2, 0.2, 0.2])   
        )
        obstacle_paths.append(obs_path)
        obstacle_positions_2d.append(obs_pos_2d)
        add_update_semantics(stage.GetPrimAtPath(obs_path), semantic_label="obstacle", type_label="class")
        num_obstacles += 1

add_reference_to_stage(usd_path=vehicle_usd, prim_path=VEHICLE_BASE_PATH)
add_reference_to_stage(usd_path=drone_usd, prim_path=DRONE_PATH)

def hide_prim(path):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid(): UsdGeom.Imageable(prim).MakeInvisible()

def show_prim(path):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid(): UsdGeom.Imageable(prim).MakeVisible()

world.reset()
hide_prim(DRONE_PATH) 

apply_semantics = lambda p, l: add_update_semantics(stage.GetPrimAtPath(p), semantic_label=l, type_label="class")
apply_semantics(VEHICLE_BASE_PATH, "vehicle") 
apply_semantics(DRONE_PATH, "drone")

drone_art = Articulation(DRONE_PATH)
vehicle_handle = XFormPrim(VEHICLE_BASE_PATH) 
world.scene.add(drone_art)

os.makedirs(output_dir, exist_ok=True)
cam_prim = UsdGeom.Camera.Define(stage, CAM_DRONE_PATH)
rp_drone = rep.create.render_product(CAM_DRONE_PATH, RESOLUTION)

with rep.new_layer():
    rep.create.light(light_type="dome", intensity=800, color=(1.0, 1.0, 1.0), name="GlobalDomeLight")
    rep.create.light(light_type="distant", intensity=2000, rotation=(-60, 0, 0), color=(1.0, 1.0, 1.0), name="SunLight")
    rep.create.light(light_type="distant", intensity=1000, rotation=(90, 0, 0), color=(1.0, 1.0, 1.0), name="BottomFill")
    
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
seg_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
for a in [rgb_annot, seg_annot, depth_annot]: a.attach(rp_drone)

def update_camera(path, eye, target):
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse()
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(m)

# ==========================================
# Main Loop
# ==========================================
print("Warm-up rendering...")
for _ in range(30): 
    world.step(render=True)
    rep.orchestrator.step()

for ep in range(NUM_EPISODES):
    ep_data = {k: [] for k in ["img", "seg", "amodal_seg", "occlusion_ratio", "depth", "pos", "quat", "target", "rel_target", "lin_vel", "ang_vel", "action"]} 

    traj_params = {
        "track_length": np.random.uniform(5.0, 8.0), 
        "track_width": np.random.uniform(3.0, 5.0),
        "base_speed": np.random.uniform(6.0, 10.0), 
        "center_offset": np.array([0.0, 0.0]) + np.random.uniform(-1.0, 1.0, size=2),
        "dir_flip": np.random.choice([1.0, -1.0]),
        "chase_dist": np.random.uniform(10.0, 14.0), 
        "chase_height": np.random.uniform(5.0, 8.0),
        "orbit_radius": np.random.uniform(2.0, 3.0), 
        "laps": np.random.uniform(1.0, 2.5)
    }
    
    start_pos, _ = get_vehicle_pose(0, traj_params)
    tracker = AggressiveTracker(init_pos=start_pos + np.array([-5.0, 3.0, 8.0]))
    current_lookat = start_pos.copy()

    # 【改良】物理推演次数增加到 1200 次
    for i in range(PHYSICS_STEPS_PER_EPISODE):
        time_t = i * DT
        target_pos, heading_deg = get_vehicle_pose(time_t, traj_params)
        q_v = R.from_euler('z', heading_deg, degrees=True).as_quat()
        vehicle_handle.set_world_pose(position=target_pos, orientation=np.array([q_v[3], q_v[0], q_v[1], q_v[2]]))
        
        v_rad = np.radians(heading_deg - VEHICLE_YAW_OFFSET)
        fwd_v = np.array([np.cos(v_rad), np.sin(v_rad), 0.0])
        # 【改良】保证轨道运算的周期对齐新的物理总步数
        p = i / (PHYSICS_STEPS_PER_EPISODE - 1)
        orbit_offset = np.array([traj_params["orbit_radius"] * np.cos(p*2*np.pi*traj_params["laps"]),
                                traj_params["orbit_radius"] * np.sin(p*2*np.pi*traj_params["laps"]),
                                1.5 * np.sin(time_t * 1.5)])
        
        ideal_drone_pos = target_pos - fwd_v * traj_params["chase_dist"] + np.array([0, 0, traj_params["chase_height"]]) + orbit_offset
        
        for obs_pos in obstacle_positions_2d:
            vec_to_obs = ideal_drone_pos[:2] - obs_pos
            dist_xy = np.linalg.norm(vec_to_obs)
            
            safe_radius = 2.5 
            if dist_xy < safe_radius:
                repulse_dir = vec_to_obs / (dist_xy + 1e-6)
                ideal_drone_pos[:2] = obs_pos + repulse_dir * safe_radius
                ideal_drone_pos[2] += 1.0 

        vec_to_car = ideal_drone_pos - target_pos
        dist = np.linalg.norm(vec_to_car)
        if dist < MIN_SAFE_DIST:
            ideal_drone_pos = target_pos + (vec_to_car / (dist + 1e-6)) * MIN_SAFE_DIST
        if ideal_drone_pos[2] < 1.0: ideal_drone_pos[2] = 1.0
        
        drone_pos, drone_lin_vel = tracker.update(ideal_drone_pos, DT)
        current_lookat = current_lookat + 0.15 * (target_pos - current_lookat)
        
        dir_vec = current_lookat - drone_pos
        fwd = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
        side = np.cross([0,0,1], fwd)
        side /= (np.linalg.norm(side) + 1e-6)
        up = np.cross(fwd, side)
        rot_m_mat = np.stack([fwd, side, up], axis=1)
        rot_m = R.from_matrix(rot_m_mat)
        q_d = rot_m.as_quat()
        
        drone_art.set_world_pose(position=drone_pos, orientation=np.array([q_d[3], q_d[0], q_d[1], q_d[2]]))
        drone_art.set_linear_velocity(drone_lin_vel)
        ang_vel = drone_art.get_angular_velocity() if drone_art.get_angular_velocity() is not None else np.zeros(3)
        
        update_camera(CAM_DRONE_PATH, eye=drone_pos + rot_m.apply([0.2, 0, 0.1]), target=current_lookat)
        
        # ====================================
        # 【核心优化】：降采样与按需渲染
        # ====================================
        should_save_data = (i % SAVE_INTERVAL == 0)
        
        # 1. 物理步进。如果当前帧不需要保存，我们直接关闭光线追踪渲染，极大提升循环速度！
        world.step(render=should_save_data)
        
        if should_save_data:
            # 只有在需要提取图像时，才触发 Replicator 的渲染管线
            rep.orchestrator.step()
            
            rgb = rgb_annot.get_data()
            seg_modal = seg_annot.get_data()
            dep = depth_annot.get_data()

            # 2. 瞬间隐藏所有遮挡物 
            for path in obstacle_paths:
                hide_prim(path)
                
            # 3. 强制仅更新渲染管线，获取无视遮挡的完美 Mask 
            world.render() 
            rep.orchestrator.step()
            
            seg_amodal = seg_annot.get_data()
            
            # 4. 恢复遮挡物可见性，准备进入下一帧 
            for path in obstacle_paths:
                show_prim(path)

            # 5. 数据解包与保存
            if rgb is not None and "data" in seg_modal and "data" in seg_amodal:
                rgb_img = rgb[:, :, :3].astype(np.uint8)
                modal_mask = seg_modal["data"]
                amodal_mask = seg_amodal["data"]
                
                vehicle_id = None
                if "info" in seg_modal and "idToLabels" in seg_modal["info"]:
                    for cid, label_info in seg_modal["info"]["idToLabels"].items():
                        if label_info.get("class") == "vehicle":
                            vehicle_id = int(cid)
                            break
                
                occ_ratio = 0.0
                if vehicle_id is not None:
                    visible_pixels = np.sum(modal_mask == vehicle_id)
                    perfect_pixels = np.sum(amodal_mask == vehicle_id)
                    if perfect_pixels > 0:
                        occ_ratio = 1.0 - (visible_pixels / perfect_pixels)
                
                ep_data["img"].append(rgb_img)
                ep_data["seg"].append(modal_mask.astype(np.uint8))
                ep_data["amodal_seg"].append(amodal_mask.astype(np.uint8)) 
                ep_data["occlusion_ratio"].append(np.float32(occ_ratio)) 
                ep_data["depth"].append(dep.astype(np.float16))
                ep_data["rel_target"].append(rot_m.inv().apply(target_pos - drone_pos).astype(np.float32))
                ep_data["action"].append(np.concatenate([drone_lin_vel, ang_vel]).astype(np.float32))
                ep_data["pos"].append(drone_pos.astype(np.float32))
                ep_data["quat"].append(q_d.astype(np.float32))
                ep_data["target"].append(target_pos.astype(np.float32))
                ep_data["lin_vel"].append(drone_lin_vel.astype(np.float32))
                ep_data["ang_vel"].append(ang_vel.astype(np.float32))

    if len(ep_data["img"]) > 0:
        save_path = os.path.join(output_dir, f"episode_{ep:05d}.npz")
        np.savez_compressed(save_path, **{k: np.array(v) for k, v in ep_data.items()})
        print(f"[Success] Episode {ep:05d} saved. (300 frames sampled across 10 seconds of physics)")

simulation_app.close()