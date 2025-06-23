import os
from typing import Union

import numpy as np
import torch
from pygltflib import Material, PbrMetallicRoughness

try:
    import smplx
except ImportError:
    raise ImportError("Please install smplx: pip install smplx")

try:
    from pygltflib import (
        ARRAY_BUFFER,
        ELEMENT_ARRAY_BUFFER,
        FLOAT,
        GLTF2,
        UNSIGNED_INT,
        UNSIGNED_SHORT,
        Accessor,
        Animation,
        AnimationChannel,
        AnimationSampler,
        Buffer,
        BufferView,
        Mesh,
        Node,
        Primitive,
        Scene,
        Skin,
    )
except ImportError:
    raise ImportError("Please install pygltflib: pip install pygltflib")

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


def axis_angle_to_quat(axis_angle):
    """axis-angle (..., 3) → quaternion (..., 4)"""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    small_angle = angle < 1e-8

    axis = np.zeros_like(axis_angle)
    axis[~small_angle[..., 0]] = axis_angle[~small_angle[..., 0]] / angle[~small_angle[..., 0]]
    axis[small_angle[..., 0]] = np.array([1.0, 0.0, 0.0])  # arbitrary

    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    quat = np.concatenate([axis * sin_half, np.cos(half_angle)], axis=-1)  # (x, y, z, w)
    return quat.astype(np.float32)


def checkerboard_geometry(
    length=12.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7],
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    num_rows = num_cols = max(2, int(length / tile_width))
    radius = float(num_rows * tile_width) / 2.0
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2

            cur_faces = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors


class MEMotion:
    """
    Stores a single motion sequence for SMPL-based animation export.

    Args:
        betas (torch.Tensor): Shape parameters, shape (B,) or (N, B).
        body_pose (torch.Tensor): Axis-angle joint rotations, shape (N, 23*3).
        global_orient (torch.Tensor): Global orientation, shape (N, 3).
        transl (torch.Tensor): Root translations, shape (N, 3).

    """

    def __init__(self, betas: torch.Tensor, body_pose: torch.Tensor, global_orient: torch.Tensor, transl: torch.Tensor):
        # Validate input types
        self._validate_input_types(betas, body_pose, global_orient, transl)

        # Validate shapes
        self._validate_shapes(betas, body_pose, global_orient, transl)  # Store data
        self.betas = betas
        self.body_pose = body_pose
        self.global_orient = global_orient
        self.transl = transl

        # Store sequence length
        self.num_frames = body_pose.shape[0]

    def _validate_input_types(self, *tensors):
        """Validate input data types"""
        for i, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Argument {i} must be torch.Tensor, got {type(tensor)}")

    def _validate_shapes(
        self, betas: torch.Tensor, body_pose: torch.Tensor, global_orient: torch.Tensor, transl: torch.Tensor
    ):
        """Validate all tensor shapes"""
        # Get sequence length
        N = body_pose.shape[0]

        # Validate body_pose shape - support 21 joints (63 dimensions)
        if body_pose.ndim != 2 or body_pose.shape[1] != 21 * 3:
            raise ValueError(f"body_pose must have shape (N, 63), got {body_pose.shape}")

        # Validate global_orient shape
        if global_orient.shape != (N, 3):
            raise ValueError(f"global_orient must have shape ({N}, 3), got {global_orient.shape}")

        # Validate transl shape
        if transl.shape != (N, 3):
            raise ValueError(f"transl must have shape ({N}, 3), got {transl.shape}")

        # Validate betas shape
        if betas.ndim == 1:
            # Single shape parameters for all frames
            pass
        elif betas.ndim == 2 and betas.shape[0] == N:
            # Different shape parameters per frame
            pass
        else:
            raise ValueError(f"betas must have shape (B,) or ({N}, B), got {betas.shape}")

        # Check if all betas are the same
        if betas.ndim == 2 and not torch.all(betas[0] == betas):
            raise ValueError(
                "All betas must be the same for SMPL-X inference. Please provide a single set of shape parameters."
            )

    @property
    def shape_params_dim(self) -> int:
        """Return shape parameters dimension"""
        if self.betas.ndim == 1:
            return self.betas.shape[0]
        else:
            return self.betas.shape[1]

    def to_device(self, device: Union[str, torch.device]) -> "MEMotion":
        """Move all tensors to specified device and return new MEMotion instance"""
        device = torch.device(device)
        return MEMotion(
            betas=self.betas.to(device),
            body_pose=self.body_pose.to(device),
            global_orient=self.global_orient.to(device),
            transl=self.transl.to(device),
        )

    def _ensure_tensor_format(self, device: Union[str, torch.device]) -> tuple:
        """Ensure all tensors are on correct device with correct type and format for SMPL"""
        device = torch.device(device)

        # Move to device and ensure float type
        betas = self.betas.float().to(device)
        body_pose = self.body_pose.float().to(device)
        global_orient = self.global_orient.float().to(device)
        transl = self.transl.float().to(device)

        # Handle betas shape for SMPL model
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).repeat(self.num_frames, 1)

        return betas, body_pose, global_orient, transl


class MEExporter:
    """
    Exports MEMotion to .glb using SMPL-X for inference and SMPL for export.

    Pipeline: SMPL-X inference → SMPL-X to SMPL conversion → GLB export

    Args:
        model_path (str): Directory containing SMPL model.
        gender (str): Gender string for the model.
        device (Union[str, torch.device]): Torch device (default 'cpu').
        conversion_matrix_path (str): Path to SMPL-X to SMPL conversion matrix.
    """

    def __init__(
        self,
        model_path: str,
        gender: str = "neutral",
        device: Union[str, torch.device] = "cpu",
        conversion_matrix_path: str = None,
        fps: int = 30,
    ):
        self.fps = fps
        self.device = torch.device(device)
        if not os.path.isdir(model_path):
            raise ValueError(f"model_path does not exist: {model_path}")
        # Create SMPL-X model for inference
        self.smplx_model = smplx.create(
            model_path,
            model_type="smplx",
            gender=gender,
            ext="npz",
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_expression=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
        ).to(self.device)

        # Create SMPL model for face data
        self.smpl_model = smplx.create(
            model_path,
            model_type="smpl",
            gender=gender,
            ext="pkl",
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        ).to(self.device)  # Load conversion matrix (SMPL-X vertices → SMPL vertices)
        if conversion_matrix_path and os.path.exists(conversion_matrix_path):
            self.smplx2smpl = torch.load(conversion_matrix_path).to(self.device)
        else:
            self.smplx2smpl = None

        # Use SMPL faces and parents for GLB export
        self.faces = self.smpl_model.faces.astype(np.int32)

    def _forward(
        self,
        betas=None,
        global_orient=None,
        transl=None,
        body_pose=None,
        left_hand_pose=None,
        right_hand_pose=None,
        expression=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        **kwargs,
    ):
        """
        Flexible SMPL-X forward pass with automatic batch size handling.

        Supports batch inference by automatically determining batch size from input tensors
        and creating default values for missing parameters.

        Args:
            betas: Shape parameters (batch_size, 10)
            global_orient: Global orientation (batch_size, 3)
            body_pose: Body pose (batch_size, 63)
            transl: Translation (batch_size, 3)
            left_hand_pose: Left hand pose (batch_size, 6) - PCA compressed
            right_hand_pose: Right hand pose (batch_size, 6) - PCA compressed
            expression: Facial expression (batch_size, 10) - optional
            jaw_pose: Jaw pose (batch_size, 3) - optional
            leye_pose: Left eye pose (batch_size, 3) - optional
            reye_pose: Right eye pose (batch_size, 3) - optional

        Returns:
            SMPL-X output with vertices and joints
        """
        device, dtype = self.smplx_model.shapedirs.device, self.smplx_model.shapedirs.dtype

        # Determine batch size from non-None inputs
        model_vars = [
            betas,
            global_orient,
            body_pose,
            transl,
            expression,
            left_hand_pose,
            right_hand_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
        ]
        batch_size = 1
        for var in model_vars:
            if var is not None:
                batch_size = max(batch_size, len(var))

        # Create default values for missing parameters
        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
            body_pose = torch.zeros([batch_size, 3 * self.smplx_model.NUM_BODY_JOINTS], dtype=dtype, device=device)
        if left_hand_pose is None:
            # SMPL-X model uses PCA compression (6 components for supermotion)
            hand_pose_dim = 6
            left_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype, device=device)
        if right_hand_pose is None:
            # SMPL-X model uses PCA compression (6 components for supermotion)
            hand_pose_dim = 6
            right_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype, device=device)
        if jaw_pose is None:
            jaw_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if leye_pose is None:
            leye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if reye_pose is None:
            reye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.smplx_model.num_expression_coeffs], dtype=dtype, device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.smplx_model.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Forward pass through SMPL-X model
        smplx_output = self.smplx_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            **kwargs,
        )

        return smplx_output

    def export(self, motion: MEMotion, filename: str):
        """
        Export a MEMotion instance as .glb file.

        Args:
            motion (MEMotion): The motion instance to export.
            filename (str): Output file path.
        """
        # Get tensors formatted for SMPL-X model
        betas, body_pose, global_orient, transl = motion._ensure_tensor_format(self.device)

        # Step 1: Use SMPL-X model for high-precision inference
        with torch.no_grad():
            smplx_vertices = self._forward(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            ).vertices  # (N, 10475, 3)
            # Get default SMPL-X vertices for initialization
            init_vertices = self._forward(
                betas=betas[:1],
                body_pose=torch.zeros_like(body_pose[:1]),
                global_orient=torch.zeros_like(global_orient[:1]),
                transl=torch.zeros_like(transl[:1]),
            ).vertices  # (1, 10475, 3)

        # Step 2: Convert SMPL-X vertices to SMPL vertices
        if self.smplx2smpl is not None:
            smpl_vertices = self._convert_smplx_to_smpl(smplx_vertices)  # (N, 6890, 3)
            init_vertices = self._convert_smplx_to_smpl(init_vertices)  # (1, 6890, 3)
        else:
            raise ValueError("SMPL-X to SMPL conversion matrix not provided. Please provide a valid conversion matrix.")

        # HACK: Align to xy plane
        # init_vertices[:, :, 2] -= init_vertices[:, :, 2].min()
        # HACK: Use initial joints as template
        init_joints = self._get_smpl_joints(init_vertices)  # (1, 24, 3)

        # Step 3: Use SMPL joint regressor to get SMPL joints
        smpl_joints = self._get_smpl_joints(smpl_vertices)  # (N, 24, 3)
        joints = smpl_joints.cpu().numpy()

        # Step 4: Build GLB with SMPL data
        vertices = init_vertices.cpu().numpy()
        gltf = self._build_gltf(vertices, init_joints, self.faces, joints, global_orient, transl, body_pose)
        gltf.save_binary(filename)
        print(f"Exported animated GLB: {filename}")
        # gltf.save_json(filename.replace(".glb", ".gltf"))

    def _convert_smplx_to_smpl(self, smplx_vertices: torch.Tensor) -> torch.Tensor:
        """Convert SMPL-X vertices to SMPL vertices using conversion matrix"""
        # smplx_vertices: (N, 10475, 3)
        # self.smplx2smpl: (6890, 10475) - sparse conversion matrix
        # Output: (N, 6890, 3)
        return torch.stack([torch.matmul(self.smplx2smpl, v) for v in smplx_vertices])

    def _get_smpl_joints(self, smpl_vertices: torch.Tensor) -> torch.Tensor:
        """Get SMPL joints from SMPL vertices using joint regressor"""
        # Use SMPL model's joint regressor
        J_regressor = self.smpl_model.J_regressor
        # smpl_vertices: (N, 6890, 3)
        # J_regressor: (24, 6890)
        # Output: (N, 24, 3)
        return torch.einsum("bvx,jv->bjx", smpl_vertices, J_regressor)

    def _build_gltf(self, vertices, init_joints, faces, joints, global_orient, transl, body_pose):
        """Build GLTF2 structure with animation data"""
        gltf = GLTF2()
        num_joints = joints.shape[1]

        # Build joint hierarchy
        parents = self.smpl_model.parents.cpu().numpy()

        # Calculate T-pose joint positions
        # J_template = (self.smpl_model.J_regressor @ self.smpl_model.v_template).cpu().numpy()  # (24, 3)
        # HACK: Use initial joints as template
        J_template = init_joints[0]

        # Prepare skinning data
        lbs_weights = self.smpl_model.lbs_weights.cpu().numpy()  # (6890, 24)
        num_vertices = lbs_weights.shape[0]

        # Prepare skinning data - convert to 4 joints per vertex format
        joints_per_vertex = []
        weights_per_vertex = []

        for v_idx in range(num_vertices):
            vertex_weights = lbs_weights[v_idx]

            # Find the 4 most influential joints
            joint_indices = np.argsort(vertex_weights)[-4:][::-1]
            joint_weights = vertex_weights[joint_indices]

            # Normalize weights to sum to 1.0
            weight_sum = joint_weights.sum()
            if weight_sum > 0:
                joint_weights = joint_weights / weight_sum
            else:
                # Fallback: assign to root joint
                joint_indices = np.array([0, 0, 0, 0], dtype=np.uint16)
                joint_weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            joints_per_vertex.extend(joint_indices.astype(np.uint16))
            weights_per_vertex.extend(joint_weights.astype(np.float32))

        num_frames = joints.shape[0]
        buffer_data = b""
        buffer_views = []
        buffer_offset = 0
        accessors = []

        # ================================
        # BUFFER 0: Vertex Position Data
        # ================================
        vert_data = vertices[0].astype(np.float32).tobytes()
        buffer_data += vert_data

        vert_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(vert_data), target=ARRAY_BUFFER)
        buffer_offset += len(vert_data)
        buffer_views.append(vert_bv)

        vert_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=vertices.shape[1],
            type="VEC3",
            max=[float(x) for x in vertices[0].max(axis=0)],
            min=[float(x) for x in vertices[0].min(axis=0)],
        )
        accessors.append(vert_acc)
        vertices_accessor_idx = len(accessors) - 1

        # ================================
        # BUFFER 1: Face Indices Data
        # ================================
        idx_data = faces.flatten().astype(np.uint32).tobytes()
        buffer_data += idx_data

        idx_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(idx_data), target=ELEMENT_ARRAY_BUFFER)
        buffer_offset += len(idx_data)
        buffer_views.append(idx_bv)

        idx_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=UNSIGNED_INT,
            count=faces.size,
            type="SCALAR",
            max=[int(faces.max())],
            min=[int(faces.min())],
        )
        accessors.append(idx_acc)
        faces_accessor_idx = len(accessors) - 1

        # ================================
        # BUFFER 2: Inverse Bind Matrices
        # ================================
        ibm = np.zeros((num_joints, 4, 4), dtype=np.float32)
        world_transforms = np.zeros((num_joints, 4, 4), dtype=np.float32)
        for joint_idx in range(num_joints):
            transform = np.eye(4, dtype=np.float32)
            if parents[joint_idx] == -1:
                transform[:3, 3] = J_template[joint_idx]
            else:
                local_pos = J_template[joint_idx] - J_template[parents[joint_idx]]
                transform[:3, 3] = local_pos
                transform = np.dot(world_transforms[parents[joint_idx]], transform)
            world_transforms[joint_idx] = transform
            ibm[joint_idx] = np.linalg.inv(transform)

        ibm_column_major = ibm.transpose(0, 2, 1).astype(np.float32)
        ibm_data = ibm_column_major.tobytes()
        buffer_data += ibm_data

        ibm_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(ibm_data))
        buffer_offset += len(ibm_data)
        buffer_views.append(ibm_bv)

        ibm_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=num_joints,
            type="MAT4",
        )
        accessors.append(ibm_acc)
        ibm_accessor_idx = len(accessors) - 1

        # ================================
        # BUFFER 3: Vertex Joint Indices
        # ================================
        joints_data = np.array(joints_per_vertex, dtype=np.uint16).tobytes()
        buffer_data += joints_data

        joints_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(joints_data), target=ARRAY_BUFFER)
        buffer_offset += len(joints_data)
        buffer_views.append(joints_bv)

        joints_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=UNSIGNED_SHORT,
            count=num_vertices,
            type="VEC4",
        )
        accessors.append(joints_acc)
        Joint_accessor_idx = len(accessors) - 1

        # ================================
        # BUFFER 4: Vertex Joint Weights
        # ================================
        weights_data = np.array(weights_per_vertex, dtype=np.float32).tobytes()
        buffer_data += weights_data

        weights_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(weights_data), target=ARRAY_BUFFER)
        buffer_offset += len(weights_data)
        buffer_views.append(weights_bv)

        weights_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=num_vertices,
            type="VEC4",
        )
        accessors.append(weights_acc)
        weights_accessor_idx = len(accessors) - 1

        # ================================
        # BUFFER 5: Animation Time Data
        # ================================
        times = np.arange(num_frames, dtype=np.float32) / self.fps
        times_data = times.tobytes()
        buffer_data += times_data

        times_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(times_data))
        buffer_offset += len(times_data)
        buffer_views.append(times_bv)

        times_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=num_frames,
            type="SCALAR",
            max=[float(times.max())],
            min=[float(times.min())],
        )
        accessors.append(times_acc)
        times_accessor_idx = len(accessors) - 1

        # ================================
        # BUFFER 6: Joint Translation Animation Data
        # ================================
        trans_accessor_idx = len(accessors)
        # trans_data = transl.cpu().numpy().astype(np.float32).tobytes()  # Use the transl
        trans_data = joints[:, 0, :].tobytes()  # Use the fitted root joint positions
        buffer_data += trans_data

        trans_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(trans_data))
        buffer_offset += len(trans_data)
        buffer_views.append(trans_bv)

        trans_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=num_frames,
            type="VEC3",
        )
        accessors.append(trans_acc)

        # ================================
        # BUFFER 7+: Joint Rotation Animation Data
        # ================================
        body_pose = body_pose.cpu().numpy().astype(np.float32).reshape(num_frames, body_pose.shape[1] // 3, 3)
        global_orient = global_orient.cpu().numpy().astype(np.float32)
        body_pose = np.concatenate([global_orient[:, np.newaxis, :], body_pose], axis=1)
        rotation_accessor_start = len(accessors)
        for joint_idx in range(body_pose.shape[1]):
            joint_rotations = []
            for frame_idx in range(num_frames):
                aa = body_pose[frame_idx, joint_idx]
                quat = axis_angle_to_quat(aa)
                joint_rotations.extend(quat.tolist())

            rot_data = np.array(joint_rotations, dtype=np.float32).tobytes()
            buffer_data += rot_data

            rot_bv = BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(rot_data))
            buffer_offset += len(rot_data)
            buffer_views.append(rot_bv)

            rot_acc = Accessor(
                bufferView=len(buffer_views) - 1,
                componentType=FLOAT,
                count=num_frames,
                type="VEC4",
            )
            accessors.append(rot_acc)

        # ================================
        # Buffer: Floor vertices
        # ================================
        floor_vertices, floor_faces, floor_vert_colors, floor_face_colors = checkerboard_geometry()
        floor_vert_data = floor_vertices.astype(np.float32).tobytes()
        buffer_data += floor_vert_data

        floor_vert_bv = BufferView(
            buffer=0,
            byteOffset=buffer_offset,
            byteLength=len(floor_vert_data),
            target=ARRAY_BUFFER,
        )
        buffer_offset += len(floor_vert_data)
        buffer_views.append(floor_vert_bv)

        floor_vert_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=floor_vertices.shape[0],
            type="VEC3",
            max=[float(x) for x in floor_vertices.max(axis=0)],
            min=[float(x) for x in floor_vertices.min(axis=0)],
        )
        accessors.append(floor_vert_acc)
        floor_vert_acc_idx = len(accessors) - 1

        # ================================
        # Buffer: Floor face indices
        # ================================
        floor_idx_data = floor_faces.flatten().astype(np.uint32).tobytes()
        buffer_data += floor_idx_data

        floor_idx_bv = BufferView(
            buffer=0,
            byteOffset=buffer_offset,
            byteLength=len(floor_idx_data),
            target=ELEMENT_ARRAY_BUFFER,
        )
        buffer_offset += len(floor_idx_data)
        buffer_views.append(floor_idx_bv)

        floor_idx_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=UNSIGNED_INT,
            count=floor_faces.size,
            type="SCALAR",
            max=[int(floor_faces.max())],
            min=[int(floor_faces.min())],
        )
        accessors.append(floor_idx_acc)
        floor_idx_acc_idx = len(accessors) - 1

        # ================================
        # Buffer: Floor vertex colors
        # ================================
        floor_color_data = floor_vert_colors.astype(np.float32).tobytes()
        buffer_data += floor_color_data
        floor_color_bv = BufferView(
            buffer=0,
            byteOffset=buffer_offset,
            byteLength=len(floor_color_data),
            target=ARRAY_BUFFER,
        )
        buffer_offset += len(floor_color_data)
        buffer_views.append(floor_color_bv)
        floor_color_acc = Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=FLOAT,
            count=floor_vertices.shape[0],
            type="VEC4",
            max=[float(x) for x in floor_vert_colors.max(axis=0)],
            min=[float(x) for x in floor_vert_colors.min(axis=0)],
        )
        accessors.append(floor_color_acc)
        floor_color_acc_idx = len(accessors) - 1

        # ================================
        # Set glTF Structure
        # ================================
        # Create joint nodes
        node_list = []
        root_joints = []

        for jid in range(num_joints):
            children = [c for c, p in enumerate(parents) if p == jid and c != jid]

            if parents[jid] == -1:
                local_translation = J_template[jid]
                root_joints.append(jid)
            else:
                local_translation = J_template[jid] - J_template[parents[jid]]

            node = Node(
                name=JOINT_NAMES[jid],
                children=children if children else [],
                translation=[float(x) for x in local_translation],
                rotation=[0.0, 0.0, 0.0, 1.0],
                scale=[1.0, 1.0, 1.0],
            )
            node_list.append(node)

        # Create mesh
        mesh = Mesh(
            name="SMPL_body_mesh",
            primitives=[
                Primitive(
                    attributes={
                        "POSITION": vertices_accessor_idx,
                        "JOINTS_0": Joint_accessor_idx,
                        "WEIGHTS_0": weights_accessor_idx,
                    },
                    indices=1,
                )
            ],
        )

        floor_material = Material(
            name="FloorMaterial",
            pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[1, 1, 1, 1]),
            doubleSided=True,
            alphaCutoff=None,
        )
        gltf.materials = [floor_material]
        floor_mesh = Mesh(
            name="checkerboard_floor",
            primitives=[
                Primitive(
                    attributes={"POSITION": floor_vert_acc_idx, "COLOR_0": floor_color_acc_idx},
                    indices=floor_idx_acc_idx,
                    material=0,
                )
            ],
        )
        gltf.meshes = [mesh, floor_mesh]

        # Create Skin
        skin = Skin(
            name="SMPL_skin",
            joints=list(range(num_joints)),
            inverseBindMatrices=ibm_accessor_idx,
        )
        gltf.skins = [skin]

        # Create Mesh Node
        mesh_node = Node(
            name="SMPL_body",
            mesh=0,
            skin=0,
        )
        node_list.append(mesh_node)
        node_list[0].children.append(len(node_list) - 1)  # Attach mesh node to root joint

        # Create Floor Mesh
        floor_node = Node(
            name="checkerboard_floor",
            mesh=1,
        )
        node_list.append(floor_node)

        gltf.nodes = node_list

        # Build Scene
        gltf.scenes = [Scene(nodes=[0, len(node_list) - 1])]
        gltf.scene = 0
        gltf.asset = {"version": "2.0", "generator": "MotionExport"}
        gltf.buffers = [Buffer(byteLength=len(buffer_data))]
        gltf.bufferViews = buffer_views
        gltf.accessors = accessors
        gltf.set_binary_blob(buffer_data)

        # ================================
        # Create Animation
        # ================================
        samplers = []
        channels = []

        # Translation sampler and channel
        sampler_trans = AnimationSampler(input=times_accessor_idx, output=trans_accessor_idx, interpolation="LINEAR")
        samplers.append(sampler_trans)

        chan_trans = AnimationChannel(sampler=len(samplers) - 1, target={"node": 0, "path": "translation"})
        channels.append(chan_trans)

        for joint_idx in range(body_pose.shape[1]):
            # Rotation sampler and channel
            rot_accessor_idx = rotation_accessor_start + joint_idx
            sampler_rot = AnimationSampler(input=times_accessor_idx, output=rot_accessor_idx, interpolation="LINEAR")
            samplers.append(sampler_rot)

            chan_rot = AnimationChannel(sampler=len(samplers) - 1, target={"node": joint_idx, "path": "rotation"})
            channels.append(chan_rot)

        # Create animation
        animation = Animation(samplers=samplers, channels=channels)
        gltf.animations = [animation]

        return gltf


if __name__ == "__main__":
    # input_pt_pathname = "demo.pt"
    # output_glb_pathname = "output.glb"
    input_pt_pathname = r"E:\Working\t2v-14B-E6-1.pt"
    output_glb_pathname = input_pt_pathname.replace(".pt", ".glb")
    fps = 16

    conversion_matrix_path = ".model/smplx2smpl_sparse.pt"

    exporter = MEExporter(
        model_path=".model",
        gender="neutral",
        device="cpu",
        conversion_matrix_path=conversion_matrix_path if os.path.exists(conversion_matrix_path) else None,
        fps=fps,
    )

    motion_data = torch.load(input_pt_pathname)["smpl_params_global"]
    motion = MEMotion(
        betas=motion_data["betas"],
        body_pose=motion_data["body_pose"],
        global_orient=motion_data["global_orient"],
        transl=motion_data["transl"],
    )
    exporter.export(motion, output_glb_pathname)
