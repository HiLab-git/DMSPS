"""
    This module is based on the Gated CRF Loss for Weakly Supervised Semantic Image Segmentation.
    The citation:
    @article{obukhov2019gated,
        author={Anton Obukhov, Stamatios Georgoulis, Dengxin Dai and Luc Van Gool},
        title={Gated CRF Loss for Weakly Supervised Semantic Image Segmentation},
        journal={arXiv:1906.04651v1},
        year={2019},
        url={https://arxiv.org/abs/1906.04651v1},
    }
    """

from random import randint
import torch
import torch.nn.functional as F
import unfoldNd

class ModelLossSemsegGatedCRF3D(torch.nn.Module):

    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input, depth_input, spacing, 
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        kernels_radius is a list, such as [5, 5, 3], which means that the kernel radius of H, W, D are 5, 5 and 3 voxels respectively.
        spacing is a list, such as [torch.tensor([0.5, 0.4]), torch.tensor([0.5, 0.4]), torch.tensor([1.5, 1.5])], which 
        means that batch_size is 2, the spacing of first volume is [0.5, 0.5, 1.5] mm and the second is [0.4, 0.4, 1.5] mm.
        """
        assert len(kernels_radius) == 3 and len(spacing) == 3
        assert y_hat_softmax.dim() == 5, 'Prediction must be a NCHWD batch'
        N, C, height_pred, width_pred, depth_pred = y_hat_softmax.shape
        assert spacing[0].shape[0] == N
        device = y_hat_softmax.device

        assert height_input % height_pred == 0 and depth_input % depth_pred == 0 and width_input % width_pred == 0 \
            and depth_input * width_pred == width_input * depth_pred, \
            f'[{depth_input}x{width_input}] !~= [{depth_pred}x{width_pred}]'
        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, depth_pred, spacing, device, custom_modality_downsamplers
        )

        denom = N * height_pred * width_pred * depth_pred

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        kernels_size = [x * 2 + 1 for x in kernels_radius]
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .reshape(N, C, kernels_size[0] * kernels_size[1] * kernels_size[2], -1).sum(dim=2, keepdim=False)
        product_kernel_x_y_hat = product_kernel_x_y_hat.reshape(N, C, height_pred, width_pred, depth_pred)

        if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            loss = -(product_kernel_x_y_hat * y_hat_softmax).sum()
            # comment out to save computation, total loss may go below 0
            loss = kernels.sum() + loss
        else:
            raise ValueError

        out = {
            'loss': loss / denom,
        }

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, depth_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool3d
        return f_down(img, (height_dst, width_dst, depth_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, depth_pred, spacing, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = ModelLossSemsegGatedCRF3D._get_mesh(
                        N, height_pred, width_pred, depth_pred, spacing, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = ModelLossSemsegGatedCRF3D._downsample(
                        feature, modality, height_pred, width_pred, depth_pred, custom_modality_downsamplers
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                ModelLossSemsegGatedCRF3D._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 5, 'Features must be a NCHWD batch'
        N, C, H, W, D = features.shape
        kernels = ModelLossSemsegGatedCRF3D._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius[0], radius[1],
                                    radius[2], :].reshape(N, C, 1, 1, 1, -1)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius[0], radius[1], radius[2], :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, D, spacing, device):
        mesh_H = torch.arange(0, H, 1, dtype=torch.float32, device=device).reshape(1, 1, H, 1, 1).repeat(N, 1, 1, W, D)
        mesh_W = torch.arange(0, W, 1, dtype=torch.float32, device=device).reshape(1, 1, 1, W, 1).repeat(N, 1, H, 1, D)
        mesh_D = torch.arange(0, D, 1, dtype=torch.float32, device=device).reshape(1, 1, 1, 1, D).repeat(N, 1, H, W, 1)
        mesh_H = mesh_H * (spacing[0].reshape(N, 1, 1, 1, 1).repeat(1, 1, H, W, D))
        mesh_W = mesh_W * (spacing[1].reshape(N, 1, 1, 1, 1).repeat(1, 1, H, W, D))
        mesh_D = mesh_D * (spacing[2].reshape(N, 1, 1, 1, 1).repeat(1, 1, H, W, D))
        return torch.cat((mesh_H, mesh_W, mesh_D), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 5, 'Unfolding requires NCHWD batch'
        N, C, H, W, D = img.shape
        diameter = [x * 2 + 1 for x in radius]
        return unfoldNd.unfoldNd(img, kernel_size=tuple(diameter), dilation=1, padding=tuple(radius), \
            stride=1).reshape(N, C, diameter[0], diameter[1], diameter[2], -1)

class ModelLossSemsegGatedCRF3D22D(torch.nn.Module):

    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input, depth_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        assert len(kernels_radius) == 3
        assert y_hat_softmax.dim() == 5, 'Prediction must be a NCHWD batch'
        N, C, height_pred, width_pred, depth_pred = y_hat_softmax.shape
        C_input = sample.shape[1]
        device = y_hat_softmax.device

        assert height_input % height_pred == 0 and depth_input % depth_pred == 0 and width_input % width_pred == 0 \
            and depth_input * width_pred == width_input * depth_pred, \
            f'[{depth_input}x{width_input}] !~= [{depth_pred}x{width_pred}]'

        #reshape [N, C, H, W, D] tensor to [N*D, C, H, W]
        y_hat_softmax = y_hat_softmax.permute(0, 4, 1, 2, 3).reshape(N * depth_pred, C, height_pred, width_pred)
        sample = sample.permute(0, 4, 1, 2, 3).reshape(N * depth_input, C_input, height_input, width_input)

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N * depth_pred, height_pred, width_pred, device, custom_modality_downsamplers
        )

        denom = N * depth_pred * height_pred * width_pred

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        kernels_size = [x * 2 + 1 for x in kernels_radius]
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .reshape(N * depth_pred, C, kernels_size[0] * kernels_size[1], -1).sum(dim=2, keepdim=False)
        product_kernel_x_y_hat = product_kernel_x_y_hat.reshape(N * depth_pred, C, height_pred, width_pred)

        if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            loss = -(product_kernel_x_y_hat * y_hat_softmax).sum()
            # comment out to save computation, total loss may go below 0
            loss = kernels.sum() + loss
        else:
            raise ValueError

        out = {
            'loss': loss / denom,
        }

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = ModelLossSemsegGatedCRF3D22D._get_mesh(
                        N, height_pred, width_pred, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = ModelLossSemsegGatedCRF3D22D._downsample(
                        feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                ModelLossSemsegGatedCRF3D22D._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        N, C, H, W = features.shape
        kernels = ModelLossSemsegGatedCRF3D22D._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius[0],
                                    radius[1], :].reshape(N, C, 1, 1, -1)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius[0], radius[1], :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, H, 1, dtype=torch.float32, device=device).reshape(
                1, 1, H, 1).repeat(N, 1, 1, W),
            torch.arange(0, W, 1, dtype=torch.float32, device=device).reshape(
                1, 1, 1, W).repeat(N, 1, H, 1),
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        N, C, H, W = img.shape
        diameter = [x * 2 + 1 for x in radius]
        return unfoldNd.unfoldNd(img, kernel_size=(diameter[0], diameter[1]), dilation=1, padding=(radius[0], radius[1]), \
            stride=1).reshape(N, C, diameter[0], diameter[1], -1)