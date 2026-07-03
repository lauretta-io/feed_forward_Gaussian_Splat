from dataclasses import dataclass
from typing import Literal


@dataclass
class EncoderVisualizerCfg:
    num_samples: int
    min_resolution: int
    export_ply: bool


@dataclass
class ReSplatGaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    exp_scale: bool
    softplus_scale: bool
    clamp_min_scale: float
    scale_detach_depth: bool
    exp_scale_bias: float
    no_rotate_sh: bool
    no_sh_mask: bool
    init_rotation_identity: bool


@dataclass
class EncoderReSplatCfg:
    name: Literal["resplat"]
    num_depth_candidates: int
    visualizer: EncoderVisualizerCfg
    gaussian_adapter: ReSplatGaussianAdapterCfg
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool
    local_mv_match: int
    gaussian_regressor_channels: int
    supervise_intermediate_depth: bool
    return_depth: bool
    sample_log_depth: bool
    bilinear_upsample_depth: bool
    no_upsample_depth: bool
    monodepth_vit_type: str
    attn_proj_channels: int | None
    knn_samples: int
    num_blocks: int
    init_use_local_knn: bool
    init_local_knn_spatial_radius: int
    init_local_knn_num_neighbor_views: int
    init_local_knn_cross_view_radius: int
    latent_downsample: int
    fixed_latent_size: bool
    init_gaussian_multiple: int
    refine_same_num_points: bool
    depth_pred_half_res: bool
    no_crop_image: bool
    num_refine: int
    train_min_refine: int
    train_max_refine: int
    num_basic_refine_blocks: int
    state_channels: int
    update_attn_proj_channels: int | None
    refine_knn_samples: int
    refine_use_local_knn: bool
    refine_local_knn_spatial_radius: int
    refine_local_knn_num_neighbor_views: int
    refine_local_knn_cross_view_radius: int
    render_error_mv_attn_blocks: int
    use_amp: bool
    pt_head_amp: bool
    pt_update_amp: bool
    use_checkpointing: bool
    init_use_checkpointing: bool
    recurrent_use_checkpointing: bool


@dataclass
class MVSplatGaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCfg
    gaussian_adapter: MVSplatGaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: list[int]
    costvolume_unet_attn_res: list[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: list[int]
    depth_unet_channel_mult: list[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    num_refine: int = 0
    use_checkpointing: bool = False
    init_use_checkpointing: bool = False


EncoderCfg = EncoderReSplatCfg | EncoderCostVolumeCfg
