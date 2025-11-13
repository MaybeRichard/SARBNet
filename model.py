from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class ChannelNorm3D(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3, 4), keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            xhat = xhat * self.weight + self.bias
        return xhat


class MaskedConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        bias: bool = True,
        dilation=1,
        groups: int = 1,
        blind_radius=0,
    ):
        kD, kH, kW = _triple(kernel_size)
        assert (
            kD % 2 == 1 and kH % 2 == 1 and kW % 2 == 1
        ), "kernel_size must be odd in all dims"
        dilD, dilH, dilW = _triple(dilation)
        padW, padH, padD = (
            (kW - 1) * dilW // 2,
            (kH - 1) * dilH // 2,
            (kD - 1) * dilD // 2,
        )
        pad = (padD, padH, padW)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=(kD, kH, kW),
            stride=1,
            padding=pad,
            dilation=(dilD, dilH, dilW),
            groups=groups,
            bias=bias,
        )

        rD, rH, rW = _triple(blind_radius)
        ker = torch.ones(kD, kH, kW, dtype=torch.float32)
        cD, cH, cW = kD // 2, kH // 2, kW // 2
        for d in range(kD):
            for h in range(kH):
                for w in range(kW):
                    if (
                        (abs(d - cD) <= rD)
                        and (abs(h - cH) <= rH)
                        and (abs(w - cW) <= rW)
                    ):
                        ker[d, h, w] = 0.0
        self.register_buffer(
            "weight_mask", ker.view(1, 1, kD, kH, kW), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_mask
        return F.conv3d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class AdaptiveBlindSpotBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, radii=(0, 1, 2)):
        super().__init__()
        self.radii = list(radii)
        self.num_branches = len(self.radii)

        base = out_channels // self.num_branches
        rem = out_channels - base * self.num_branches
        ch_list = [base] * self.num_branches
        ch_list[-1] += rem

        branches = []
        for r, ch in zip(self.radii, ch_list):
            ks = (3, 2 * r + 3, 2 * r + 3)
            branches.append(
                MaskedConv3d(
                    in_channels,
                    ch,
                    kernel_size=ks,
                    blind_radius=(0, r, r),
                    bias=True,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.gate = nn.Conv3d(
            in_channels,
            self.num_branches,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        weights = self.softmax(logits)
        outs = []
        for i, conv in enumerate(self.branches):
            y = conv(x)
            w = weights[:, i : i + 1]
            outs.append(y * w)
        return torch.cat(outs, dim=1)


class SpectralAmplitudeGating3D(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float = 0.2,
        reduction: int = 4,
        eps: float = 1e-6,
        fft_mode: str = "3d",
    ):
        super().__init__()
        assert reduction >= 1, "reduction must be >= 1"
        self.alpha = alpha
        self.eps = eps
        self.fft_mode = fft_mode

        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        if self.fft_mode == "3d":
            Z = torch.fft.rfftn(x, dim=(2, 3, 4), norm="ortho")
            A = torch.abs(Z)
            P = Z / (A + self.eps)

            A_agg = A.mean(dim=(2, 3, 4), keepdim=True)
            w = self.gate(torch.log1p(A_agg))
            A2 = A * (1.0 + self.alpha * w)

            Z2 = P * A2
            y = torch.fft.irfftn(Z2, s=(D, H, W), dim=(2, 3, 4), norm="ortho").real
            return y

        elif self.fft_mode == "2dxy":
            Z = torch.fft.rfftn(x, dim=(3, 4), norm="ortho")
            A = torch.abs(Z)
            P = Z / (A + self.eps)

            A_agg = A.mean(dim=(2, 3, 4), keepdim=True)
            w = self.gate(torch.log1p(A_agg))
            A2 = A * (1.0 + self.alpha * w)

            Z2 = P * A2
            y = torch.fft.irfftn(Z2, s=(H, W), dim=(3, 4), norm="ortho").real
            return y
        else:
            raise ValueError(f"Invalid fft_mode: {self.fft_mode}")


class AdaptiveMaskedMultiKernelConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_kernels: List[Tuple[int, int, int]],
        branch_radii: List[Tuple[int, int, int]],
    ):
        super().__init__()
        assert (
            len(branch_kernels) == len(branch_radii) and len(branch_kernels) > 0
        ), "branch_kernels and branch_radii must have same non-zero length"
        self.num_branches = len(branch_kernels)

        base = out_channels // self.num_branches
        rem = out_channels - base * self.num_branches
        ch_list = [base] * self.num_branches
        ch_list[-1] += rem

        branches = []
        for ks, rad, ch in zip(branch_kernels, branch_radii, ch_list):
            branches.append(
                MaskedConv3d(
                    in_channels,
                    ch,
                    kernel_size=ks,
                    blind_radius=rad,
                    bias=True,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.gate = nn.Conv3d(
            in_channels,
            self.num_branches,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        weights = self.softmax(logits)
        outs = []
        for i, conv in enumerate(self.branches):
            y = conv(x)
            w = weights[:, i : i + 1]
            outs.append(y * w)
        return torch.cat(outs, dim=1)


class GlobalChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.conv1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3, 4), keepdim=True)
        s = self.conv1(s)
        s = self.relu(s)
        s = self.conv2(s)
        s = self.sigmoid(s)
        return x * s


class SimpleGatedNonlinearity3D(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.pw = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.out = nn.Conv3d(
            hidden // 2 if hidden % 2 == 0 else channels,
            channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw(x)

        c = h.shape[1]
        c1 = c // 2
        x1, x2 = h[:, :c1], h[:, c1:]
        if x2.shape[1] != x1.shape[1]:
            pad = x1.shape[1] - x2.shape[1]
            x2 = torch.cat([x2, x2[:, :pad]], dim=1)
        g = x1 * x2
        y = self.out(g)
        return y


class SRBlock3DMasked(nn.Module):
    def __init__(
        self,
        channels: int,
        branch_kernels: List[Tuple[int, int, int]],
        branch_radii: List[Tuple[int, int, int]],
        sca_reduction: int = 4,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        self.norm1 = ChannelNorm3D(channels)
        self.spatial_mixer = AdaptiveMaskedMultiKernelConv3d(
            in_channels=channels,
            out_channels=channels,
            branch_kernels=branch_kernels,
            branch_radii=branch_radii,
        )
        self.sca = GlobalChannelAttention3D(channels, reduction=sca_reduction)
        self.norm2 = ChannelNorm3D(channels)
        self.ffn = SimpleGatedNonlinearity3D(channels, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.spatial_mixer(y)
        y = self.sca(y)
        x = x + y
        z = self.norm2(x)
        z = self.ffn(z)
        x = x + z
        return x


class BlindSpotConv3DBlockSAG(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck: bool = False,
        conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
        conv_blind_radius: Tuple[int, int, int] = (0, 0, 0),
        conv_dilation: Tuple[int, int, int] = (1, 1, 1),
        use_adaptive: bool = False,
        adaptive_radii=(0, 1, 2),
        use_fre_mlp: bool = True,
        fremlp_alpha: float = 0.2,
        fremlp_reduction: int = 4,
        fremlp_fft_mode: str = "3d",
    ):
        super().__init__()
        self.bottleneck = bottleneck
        self.use_adaptive = use_adaptive
        self.use_fre_mlp = use_fre_mlp

        if use_adaptive:
            self.conv1 = AdaptiveBlindSpotBlock3D(
                in_channels,
                out_channels // 2,
                radii=adaptive_radii,
            )
        else:
            self.conv1 = MaskedConv3d(
                in_channels,
                out_channels // 2,
                kernel_size=conv_kernel_size,
                blind_radius=conv_blind_radius,
                dilation=conv_dilation,
                bias=True,
            )
        self.bn1 = ChannelNorm3D(out_channels // 2)

        if use_adaptive:
            self.conv2 = AdaptiveBlindSpotBlock3D(
                out_channels // 2,
                out_channels,
                radii=adaptive_radii,
            )
        else:
            self.conv2 = MaskedConv3d(
                out_channels // 2,
                out_channels,
                kernel_size=conv_kernel_size,
                blind_radius=conv_blind_radius,
                dilation=conv_dilation,
                bias=True,
            )
        self.bn2 = ChannelNorm3D(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

        if self.use_fre_mlp:
            self.fre_norm = ChannelNorm3D(out_channels)
            self.fre_mlp = SpectralAmplitudeGating3D(
                channels=out_channels,
                alpha=fremlp_alpha,
                reduction=fremlp_reduction,
                fft_mode=fremlp_fft_mode,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.relu(self.bn2(self.conv2(res)))

        if self.use_fre_mlp:
            z = self.fre_norm(res)
            res = res + self.fre_mlp(z)

        out = self.pooling(res) if not self.bottleneck else res
        return out, res


class BlindSpotUpConv3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        res_channels: int = 0,
        last_layer: bool = False,
        num_classes: Optional[int] = None,
        conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
        conv_blind_radius: Tuple[int, int, int] = (0, 0, 0),
        conv_dilation: Tuple[int, int, int] = (1, 1, 1),
        use_adaptive: bool = False,
        adaptive_radii=(0, 1, 2),
        use_srblock: bool = False,
        srblock_branch_kernels: Optional[List[Tuple[int, int, int]]] = None,
        srblock_branch_radii: Optional[List[Tuple[int, int, int]]] = None,
        srblock_sca_reduction: int = 4,
        srblock_ffn_expansion: int = 2,
    ):
        super().__init__()
        assert (not last_layer and num_classes is None) or (
            last_layer and num_classes is not None
        )

        self.upconv1 = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = ChannelNorm3D(in_channels // 2)

        if use_adaptive:
            self.conv1 = AdaptiveBlindSpotBlock3D(
                in_channels // 2 + res_channels,
                in_channels // 2,
                radii=adaptive_radii,
            )
            self.conv2 = AdaptiveBlindSpotBlock3D(
                in_channels // 2,
                in_channels // 2,
                radii=adaptive_radii,
            )
        else:
            self.conv1 = MaskedConv3d(
                in_channels // 2 + res_channels,
                in_channels // 2,
                kernel_size=conv_kernel_size,
                blind_radius=conv_blind_radius,
                dilation=conv_dilation,
                bias=True,
            )
            self.conv2 = MaskedConv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=conv_kernel_size,
                blind_radius=conv_blind_radius,
                dilation=conv_dilation,
                bias=True,
            )
        self.last_layer = last_layer

        if last_layer:
            self.conv3 = MaskedConv3d(
                in_channels // 2,
                num_classes,
                kernel_size=conv_kernel_size,
                blind_radius=conv_blind_radius,
                dilation=conv_dilation,
                bias=True,
            )

        self.use_srblock = use_srblock
        if self.use_srblock:
            if srblock_branch_kernels is None:
                srblock_branch_kernels = [(3, 3, 3), (3, 5, 5), (3, 7, 7)]
            if srblock_branch_radii is None:
                srblock_branch_radii = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
            self.srblock = SRBlock3DMasked(
                channels=in_channels // 2,
                branch_kernels=srblock_branch_kernels,
                branch_radii=srblock_branch_radii,
                sca_reduction=srblock_sca_reduction,
                ffn_expansion=srblock_ffn_expansion,
            )

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.upconv1(x)
        if residual is not None:
            x = torch.cat((x, residual), dim=1)
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        if getattr(self, "use_srblock", False):
            x = self.srblock(x)
        if self.last_layer:
            x = self.conv3(x)
        return x


class BlindSpotUNet3D_SAG(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        level_channels: List[int] = [64, 128, 256],
        bottleneck_channel: int = 512,
        encoder_kernel_sizes: Optional[List[Tuple[int, int, int]]] = None,
        encoder_blind_radii: Optional[List[Tuple[int, int, int]]] = None,
        decoder_kernel_sizes: Optional[List[Tuple[int, int, int]]] = None,
        decoder_blind_radii: Optional[List[Tuple[int, int, int]]] = None,
        conv_dilation: Tuple[int, int, int] = (1, 1, 1),
        encoder_adaptive: Optional[List[bool]] = None,
        decoder_adaptive: Optional[List[bool]] = None,
        adaptive_radii=(0, 1, 2),
        encoder_use_fremlp: Optional[List[bool]] = None,
        fremlp_alpha: float = 0.2,
        fremlp_reduction: int = 4,
        fremlp_fft_mode: str = "3d",
        decoder_use_srblock: Optional[List[bool]] = None,
        srblock_branch_kernels: Optional[List[Tuple[int, int, int]]] = None,
        srblock_branch_radii: Optional[List[Tuple[int, int, int]]] = None,
        srblock_sca_reduction: int = 4,
        srblock_ffn_expansion: int = 2,
    ):
        super().__init__()
        l1, l2, l3 = level_channels[0], level_channels[1], level_channels[2]

        if encoder_kernel_sizes is None:
            encoder_kernel_sizes = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
        if encoder_blind_radii is None:
            encoder_blind_radii = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        if decoder_kernel_sizes is None:
            decoder_kernel_sizes = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
        if decoder_blind_radii is None:
            decoder_blind_radii = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        if encoder_adaptive is None:
            encoder_adaptive = [False, True, True]
        if decoder_adaptive is None:
            decoder_adaptive = [True, True, False]
        if encoder_use_fremlp is None:
            encoder_use_fremlp = [True, True, False]
        if decoder_use_srblock is None:
            decoder_use_srblock = [False, True, True]

        self.a_block1 = BlindSpotConv3DBlockSAG(
            in_channels=in_channels,
            out_channels=l1,
            conv_kernel_size=encoder_kernel_sizes[0],
            conv_blind_radius=encoder_blind_radii[0],
            conv_dilation=conv_dilation,
            use_adaptive=encoder_adaptive[0],
            adaptive_radii=adaptive_radii,
            use_fre_mlp=encoder_use_fremlp[0],
            fremlp_alpha=fremlp_alpha,
            fremlp_reduction=fremlp_reduction,
            fremlp_fft_mode=fremlp_fft_mode,
        )
        self.a_block2 = BlindSpotConv3DBlockSAG(
            in_channels=l1,
            out_channels=l2,
            conv_kernel_size=encoder_kernel_sizes[1],
            conv_blind_radius=encoder_blind_radii[1],
            conv_dilation=conv_dilation,
            use_adaptive=encoder_adaptive[1],
            adaptive_radii=adaptive_radii,
            use_fre_mlp=encoder_use_fremlp[1],
            fremlp_alpha=fremlp_alpha,
            fremlp_reduction=fremlp_reduction,
            fremlp_fft_mode=fremlp_fft_mode,
        )
        self.a_block3 = BlindSpotConv3DBlockSAG(
            in_channels=l2,
            out_channels=l3,
            conv_kernel_size=encoder_kernel_sizes[2],
            conv_blind_radius=encoder_blind_radii[2],
            conv_dilation=conv_dilation,
            use_adaptive=encoder_adaptive[2],
            adaptive_radii=adaptive_radii,
            use_fre_mlp=encoder_use_fremlp[2],
            fremlp_alpha=fremlp_alpha,
            fremlp_reduction=fremlp_reduction,
            fremlp_fft_mode=fremlp_fft_mode,
        )
        self.bottleNeck = BlindSpotConv3DBlockSAG(
            in_channels=l3,
            out_channels=bottleneck_channel,
            bottleneck=True,
            conv_kernel_size=(3, 3, 3),
            conv_blind_radius=(0, 0, 0),
            conv_dilation=conv_dilation,
            use_adaptive=False,
            adaptive_radii=adaptive_radii,
            use_fre_mlp=False,
            fremlp_alpha=fremlp_alpha,
            fremlp_reduction=fremlp_reduction,
            fremlp_fft_mode=fremlp_fft_mode,
        )

        self.s_block3 = BlindSpotUpConv3DBlock(
            in_channels=bottleneck_channel,
            res_channels=l3,
            conv_kernel_size=decoder_kernel_sizes[0],
            conv_blind_radius=decoder_blind_radii[0],
            conv_dilation=conv_dilation,
            use_adaptive=decoder_adaptive[0],
            adaptive_radii=adaptive_radii,
            use_srblock=decoder_use_srblock[0],
            srblock_branch_kernels=srblock_branch_kernels,
            srblock_branch_radii=srblock_branch_radii,
            srblock_sca_reduction=srblock_sca_reduction,
            srblock_ffn_expansion=srblock_ffn_expansion,
        )
        self.s_block2 = BlindSpotUpConv3DBlock(
            in_channels=l3,
            res_channels=l2,
            conv_kernel_size=decoder_kernel_sizes[1],
            conv_blind_radius=decoder_blind_radii[1],
            conv_dilation=conv_dilation,
            use_adaptive=decoder_adaptive[1],
            adaptive_radii=adaptive_radii,
            use_srblock=decoder_use_srblock[1],
            srblock_branch_kernels=srblock_branch_kernels,
            srblock_branch_radii=srblock_branch_radii,
            srblock_sca_reduction=srblock_sca_reduction,
            srblock_ffn_expansion=srblock_ffn_expansion,
        )
        self.s_block1 = BlindSpotUpConv3DBlock(
            in_channels=l2,
            res_channels=l1,
            num_classes=num_classes,
            last_layer=True,
            conv_kernel_size=decoder_kernel_sizes[2],
            conv_blind_radius=decoder_blind_radii[2],
            conv_dilation=conv_dilation,
            use_adaptive=decoder_adaptive[2],
            adaptive_radii=adaptive_radii,
            use_srblock=decoder_use_srblock[2],
            srblock_branch_kernels=srblock_branch_kernels,
            srblock_branch_radii=srblock_branch_radii,
            srblock_sca_reduction=srblock_sca_reduction,
            srblock_ffn_expansion=srblock_ffn_expansion,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, r1 = self.a_block1(x)
        out, r2 = self.a_block2(out)
        out, r3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        out = self.s_block3(out, r3)
        out = self.s_block2(out, r2)
        out = self.s_block1(out, r1)
        return out


BlindSpotUNet3D_FreMLP = BlindSpotUNet3D_SAG


def create_model(
    model_type: str, in_channels: int = 1, num_classes: int = 1, **kwargs
) -> nn.Module:
    model_type_l = model_type.lower()
    if model_type_l in (
        "blindspotunet3d_sag",
        "blindspotunet3d_fremlp",
        "blindspotunet3d-fremlp",
    ):
        return BlindSpotUNet3D_SAG(in_channels, num_classes, **kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")


def get_model_info(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("BlindSpotUNet3D_SAG quick test (alias: BlindSpotUNet3D_FreMLP)")
    print("=" * 80)
    x = torch.randn(1, 1, 8, 128, 128)
    model = create_model(
        "BlindSpotUNet3D_FreMLP",
        in_channels=1,
        num_classes=1,
        encoder_use_fremlp=[True, True, False],
        fremlp_fft_mode="3d",
    )
    model.eval()
    with torch.no_grad():
        y = model(x)
    info = get_model_info(model)
    print(
        f"Params: {info['total_parameters']:,}  Size: {info['model_size_mb']:.2f} MB  Output: {y.shape}"
    )
