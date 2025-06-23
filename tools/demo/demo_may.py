import argparse
import shutil
from pathlib import Path

import hydra
import nanoid
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from may import MayTask
from motion_export import MEExporter, MEMotion
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.geo.hmr_cam import convert_K_to_K4, create_camera_sensor, estimate_K, get_bbx_xys_from_xyxy
from hmr4d.utils.geo_transform import apply_T_on_points, compute_cam_angvel, compute_T_ayfz2ay
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_video_reader,
    get_writer,
    merge_videos_horizontal,
    read_video_np,
    save_video,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values."
        "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
        "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc="Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get visual odometry results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:  # DPVO
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:  # DPVO
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc="Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc="Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


class GVHMRMayTask(MayTask):
    """
    GVHMR May Task for running the demo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Setup Config
        args = self._get_argparse()
        self.cfg = self._get_cfg(args)

        # Load Exporter
        self.exporter = MEExporter(
            model_path="inputs/checkpoints/body_models",
            gender="neutral",
            device="cpu",
            conversion_matrix_path="hmr4d/utils/body_model/smplx2smpl_sparse.pt",
            fps=30,
        )

        # Load Model
        model: DemoPL = hydra.utils.instantiate(self.cfg.model, _recursive_=False)
        model.load_pretrained_model(self.cfg.ckpt_path)
        self.model = model.eval().cuda()

        Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
        Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")

    def _get_argparse(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
        parser.add_argument("--output_root", type=str, default="outputs/may", help="by default to outputs/may")
        parser.add_argument("-s", "--static_cam", action="store_true", default=True, help="If true, skip DPVO")
        parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
        parser.add_argument(
            "--f_mm",
            type=int,
            default=None,
            help="Focal length of fullframe camera in mm. Leave it as None to use default values."
            "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
            "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
        )
        parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
        return parser.parse_args()

    def _get_cfg(self, args):
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            overrides = [
                "video_name=video_path",
                f"static_cam={args.static_cam}",
                f"verbose={args.verbose}",
                f"use_dpvo={args.use_dpvo}",
            ]
            if args.f_mm is not None:
                overrides.append(f"f_mm={args.f_mm}")
            if args.output_root is not None:
                overrides.append(f"output_root={args.output_root}")
            register_store_gvhmr()
            cfg = compose(config_name="demo", overrides=overrides)
        return cfg

    def _export_to_glb(self, pred: dict, output_pathname: str):
        motion_data = pred["smpl_params_global"]
        motion = MEMotion(
            betas=motion_data["betas"],
            body_pose=motion_data["body_pose"],
            global_orient=motion_data["global_orient"],
            transl=motion_data["transl"],
        )
        self.exporter.export(motion, output_pathname)

    def run(self, payload: dict):
        # Check video
        video_path = Path(payload["video_pathname"])
        assert video_path.exists(), f"Video not found at {video_path}"
        length, width, height = get_video_lwh(video_path)
        Log.info(f"[Input]: {video_path}")
        Log.info(f"(L, W, H) = ({length}, {width}, {height})")
        result_payload = dict(
            video_pathname=str(video_path.absolute()), video_length=length, video_width=width, video_height=height
        )

        # Set video path
        cfg = self.cfg
        cfg.video_name = f"{video_path.stem}-{nanoid.generate()}"
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

        # Copy raw-input-video to video_path
        Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
        if not Path(cfg.video_path).exists() or length != get_video_lwh(cfg.video_path)[0]:
            reader = get_video_reader(video_path)
            writer = get_writer(cfg.video_path, fps=30, crf=CRF)
            for img in tqdm(reader, total=length, desc="Copy"):
                writer.write_frame(img)
            writer.close()
            reader.close()

        # Prepare data
        run_preprocess(cfg)
        data = load_data_dict(cfg)

        # Predict
        pred = self.model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)

        # Save prediction
        torch.save(pred, cfg.paths.hmr4d_results)
        result_payload["gvhmr_result_pt_pathname"] = cfg.paths.hmr4d_results

        # Save motion
        glb_pathname = cfg.paths.hmr4d_results.replace(".pt", ".glb")
        self._export_to_glb(pred, glb_pathname)
        result_payload["gvhmr_result_glb_pathname"] = glb_pathname

        # Render videos
        # render_incam(cfg)
        # result_payload["gvhmr_incam_video_pathname"] = cfg.paths.incam_video
        # render_global(cfg)
        # result_payload["gvhmr_global_video_pathname"] = cfg.paths.global_video
        # merge_videos_horizontal([cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video)
        # result_payload["gvhmr_incam_global_horiz_video_pathname"] = cfg.paths.incam_global_horiz_video

        self.save_result(
            result=result_payload,
            success=True,
        )

        # Cleanup
        shutil.rmtree(cfg.preprocess_dir)
        Path(cfg.video_path).unlink(missing_ok=True)


if __name__ == "__main__":
    task = GVHMRMayTask(task_id="6TM7Sikt1pyUGieAMhutT")
    # task.run(dict(video_pathname="/data/jiahao/projects/GVHMR/inputs/data/2025-06-05/t2v-14B-E5-1.mp4"))
    task.run_server()
