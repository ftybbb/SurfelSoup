import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models import *
import open3d, os

class get_model(nn.Module):
    def __init__(self, layers=5, channels=32, surfel_start=2, feature_to=2, octree_end=1, out_channels=16, hyper=True, adaptive_quant=False, mode='2', bar=False, selection_mode='gumbel'):
        super(get_model, self).__init__()
        self.layers = layers
        self.surfel_start = surfel_start
        self.feature_to = feature_to
        self.bar = bar
        self.mode = mode
        self.adaptive_quant = adaptive_quant
        self.octree_end = octree_end
        self.selection_mode = selection_mode
        self.channels = channels
        self.enc = torch.nn.ModuleList([DownSampler(in_channels=1, channels=channels, out_channels=channels,
                                kernel_size=3, block_layers0=4, block_layers1=4)])
        for i in range(layers-1):
            self.enc.append(DownSampler(in_channels=channels, channels=channels, out_channels=channels,
                                kernel_size=3, block_layers0=4, block_layers1=4))
        self.dec = torch.nn.ModuleList([nn.Identity() for i in range(octree_end)])
        for i in range(octree_end, layers):
            self.dec.append(UpSampler8StageP(in_channels=channels, channels=32, kernel_size=3, block_layers=4, adaptor=False))

        # We tried to add multi-layer latent below f^L (f^L+1, f^L+2), but they are all-zero after training.
        self.contextDetach = nn.ModuleList([nn.Identity() for i in range(feature_to)])
        for i in range(feature_to, layers):
            self.contextDetach.append(ContextDetach(channels, channels, out_channels, mode=mode))

        self.contextRestore = nn.ModuleList([nn.Identity() for i in range(feature_to)])
        for i in range(feature_to, layers):
            self.contextRestore.append(ContextRestore(out_channels, channels, mode=mode))
            
        self.entropy = nn.ModuleList([nn.Identity() for i in range(feature_to)])
        for i in range(feature_to, layers):
            self.entropy.append(HyperPriorMultiStage(out_channels, channels, out_channels, resnet=ResNet, block_layers=1, ctxSize=out_channels+channels if i!=layers-1 else 0, hyper=hyper, adaptive_quant=True, stage=8))

        self.decision = nn.ModuleList([nn.Identity() for i in range(octree_end)])
        for i in range(octree_end, surfel_start+1):
            self.decision.append(nn.Sequential(
                make_layer(ResNet, 3, 64+(64 if self.adaptive_quant else 0), kernel_size=1),
                ME.MinkowskiConvolution(in_channels=64+(64 if self.adaptive_quant else 0), out_channels=2, kernel_size=1, stride=1, bias=True, dimension=3)
            ))

        self.GSdec = nn.ModuleList()
        self.GSScaleAdapt = nn.ModuleList()
        self.GSOpacity = nn.ModuleList()
        for i in range(layers):
            self.GSdec.append(nn.Sequential(
                make_layer(ResNet, 3, 64, kernel_size=1),
                ME.MinkowskiConvolution(in_channels=64, out_channels=11, kernel_size=1, stride=1, bias=True, dimension=3),
            ))
        self.upsample = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3, expand_coordinates=True)

    def forward(self, f1, f2, device, epoch=99999, show_motion=False, showdir='./show', frame_idx=999, keep_adaptive=False, rho=1.0, return_bce=True, return_recon=False, lamb=10, step=10000, mode_sr=False, yuv=False, normal=None):
        ''' initialization '''
        # make sure f2 has no color input
        f2 = ME.SparseTensor(torch.ones_like(f2.F[:, :1]), coordinates=f2.C, tensor_stride=f2.tensor_stride[0], device=f2.device)
        num_points = f2.C.size()[0]
        ys = [0 for i in range(self.layers)]
        ys[0] = f2
        latents = [0 for i in range(self.layers)]
        quant_latents = [0 for i in range(self.layers)]
        octree_bits_list = torch.zeros(self.layers, dtype=torch.float32, device=device)
        latent_bits_list = torch.zeros(self.layers, dtype=torch.float32, device=device)
        bce_list = torch.zeros(self.layers, dtype=torch.float32, device=device)
        l_list = torch.tensor([], dtype=torch.long, device=f2.device)
        c_list, p_list, idx_list, surf_c_list, surf_l_list, surf_m_list, surf_v_list, surf_q_list, surf_bt_list = [], [], [], [], [], [], [], [], []
        quant_latent_sum, prob_sum, prob_surf_sum = [], [], []

        ######## encoder ########
        for i in range(1, self.layers):
            ys[i] = self.enc[i-1](ys[i-1])

        downsample = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        # for code consistency, we downsample the last layer.
        ysd = downsample(ys[-1])
        # prob: probability of each node being octree node; _ste: straight-through estimator for prob (see P-SOPA); _up: upsampled version of prob, the upsample here is fixed-coordinate because we will losslessly code the octants.
        prob = ME.SparseTensor(torch.ones(ysd.C.size()[0], 1, dtype=torch.float32, device=ysd.device), coordinate_manager=ysd.coordinate_manager, coordinate_map_key=ysd.coordinate_map_key)
        prob_up = ME.SparseTensor(torch.ones(ys[-1].C.size()[0], 1, dtype=torch.float32, device=ys[-1].device), coordinate_manager=ys[-1].coordinate_manager, coordinate_map_key=ys[-1].coordinate_map_key, device=ys[-1].device)
        prob_ste = ME.SparseTensor(torch.ones_like(prob.F, device=prob.device), coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key, device=prob.device)
        prob_ste_up = ME.SparseTensor(torch.ones(ys[-1].C.size()[0], 1, dtype=torch.float32, device=ys[-1].device), coordinate_manager=ys[-1].coordinate_manager, coordinate_map_key=ys[-1].coordinate_map_key, device=ys[-1].device)

        ys_pred = ME.SparseTensor(torch.zeros_like(ys[-1].F), coordinate_manager=ys[-1].coordinate_manager, coordinate_map_key=ys[-1].coordinate_map_key, device=ys[-1].device)
        topkp1_list, topkp1_idx_list = [], []
        surf_counter = 0

        for i in range(self.layers-1, 0, -1):

            ############ latent compression ############
            if i>=self.feature_to:
                latents[i] = self.contextDetach[i](ys[i], ys_pred)
                if i==self.layers-1:
                    # last layer has no context because it's the first to be decoded
                    ctx = None
                else:
                    ctx = [quant_latents[i+1], ys_pred]
                compressionOutput = self.entropy[i](latents[i], training=self.training, gaussion=True, require_quant=True, ctx=ctx, bar=self.bar)
                quant_latent_sum.append(compressionOutput['x_ada'].F.round().abs().sum().item())
                latent_bits = compressionOutput['b_x']
                latent_bits_list[i] = latent_bits.sum()
                quant_latents[i] = compressionOutput['x_hat'] if not self.bar else compressionOutput['x_bar']
                quant_latents[i] = ME.SparseTensor(quant_latents[i].F, coordinate_manager=latents[i].coordinate_manager, coordinate_map_key=latents[i].coordinate_map_key, device=latents[i].device)
                ys_rec = self.contextRestore[i](quant_latents[i], ys_pred)
            else:
                ys_rec = ys_pred

            ##### Tree Decision #####
            upsampleNonExpand = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3, expand_coordinates=False)
            prob = upsampleNonExpand(prob)
            decision_input_F = ys_rec.F
            decision_input = ME.SparseTensor(torch.cat([decision_input_F, torch.zeros_like(decision_input_F)], dim=-1), coordinate_manager=ys_rec.coordinate_manager, coordinate_map_key=ys_rec.coordinate_map_key, device=ys_rec.device)
            if i<=self.octree_end:
                # for any layer finer than octree_end, nodes will be forced to be surfel nodes. This option is for training speedup because finer layers' lossless compression is costly and slow.
                prob_surf = prob
                prob = ME.SparseTensor(torch.zeros_like(prob.F), coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key, device=prob.device)
            elif i<=self.surfel_start:
                prob_condition = self.decision[i](decision_input)
                # for any layer coarser than surfel_start, nodes will be forced to be octree nodes. This is because the lower layer's octree lossless compression costs not too much bit rates.
                prob_condition = sort_sparse_tensor(prob_condition, prob).F
                if self.selection_mode == 'softmax':
                    prob_condition = torch.softmax(prob_condition, dim=1)[:, :1]
                elif self.selection_mode == 'gumbel':
                    tau = max(0.1, math.exp(-0.00035*step)*1.5)
                    prob_condition = F.gumbel_softmax(
                        prob_condition, tau=tau, hard=False, dim=-1
                    )[:, :1]
                if self.selection_mode == 'fixed':
                    # if selectionMode is fixed, every layer will have a fixed probability of being surfel nodes (note that the decision is still stocastic, but the probability fixed).
                    prob_surf = ME.SparseTensor(torch.ones_like(prob_condition.F[:, :1])*(1/(self.surfel_start-self.octree_end+1)), coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key)
                    prob = prob-prob_surf
                else:
                    prob_surf = prob*(1-prob_condition)
                    prob = prob*prob_condition

            # P-SOPA STE
            prob_ste_F = torch.rand_like(prob.F, device=prob.device)<=prob.F
            prob_ste_F = prob_ste_F.float()
            prob_ste_F = (prob_ste_F-prob.F).detach()+prob.F
            prob_ste = ME.SparseTensor(prob_ste_F, coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key, device=prob.device)

            if i<=self.surfel_start:
                prob_sum.append((prob.F.sum()/ys[i].size()[0]).item())
                prob_surf_sum.append((prob_surf.F.sum()/ys[i].size()[0]).item())
                


            ######### octree reconstruction (for next layer) ##########
            if i>self.octree_end:
                octree_out_set = self.dec[i](ys_rec, ys[i-1], training=self.training, prob=prob, prob_ste=prob_ste)
                ys_pred = octree_out_set['out']
                prob_up, prob_ste_up = octree_out_set['prob_up'], octree_out_set['prob_ste_up']
                out_cls, target = octree_out_set['out_cls_list'], octree_out_set['target_list']
                out_cls = ME.SparseTensor(torch.cat([out_cls[i].F for i in range(len(out_cls))], dim=0), coordinates=torch.cat([out_cls[i].C for i in range(len(out_cls))], dim=0), tensor_stride=ys[i-1].tensor_stride[0])
                target = torch.cat(target, dim=0)
                bcefunc = nn.BCEWithLogitsLoss(reduction='none')
                
                octree_bits = bcefunc(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))/math.log(2.0)
                octree_bits = ME.SparseTensor(octree_bits.unsqueeze(-1), coordinates=out_cls.C, tensor_stride=ys[i-1].tensor_stride[0])
                sumPooling = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=3)
                octree_bits = sumPooling(octree_bits)
                octree_bits = sort_sparse_tensor(octree_bits, prob)
                octree_bits = octree_bits*prob.F
                octree_bits_list[i] = octree_bits.F.sum()

            ######## Surfel Reconstruction ########
            if i<=self.surfel_start:
                surfel_input = ME.SparseTensor(torch.cat([ys_rec.F, torch.zeros_like(ys_rec.F)], dim=-1), coordinate_manager=ys_rec.coordinate_manager, coordinate_map_key=ys_rec.coordinate_map_key, device=ys_rec.device)
                params = self.GSdec[i](surfel_input)
                stride = params.tensor_stride[0]
                params_F = params.F
                idxs = ME.SparseTensor(
                    torch.arange(params.C.size()[0], device=params.device, dtype=torch.float32).unsqueeze(1) + surf_counter,
                    coordinate_map_key=params.coordinate_map_key,
                    coordinate_manager=params.coordinate_manager,
                    device=params.device,
                )
                surf_counter += params.C.size()[0]
                means = ME.SparseTensor(params_F[:, :3], coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                scales_F = torch.exp(params_F[:, 3:6].clamp(min=-8, max=8))
                scales = ME.SparseTensor(scales_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                quats_F = params_F[:, 6:10]
                quats_F = quats_F / (quats_F.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)
                quats = ME.SparseTensor(quats_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                rotmat_F = quat2rotmat(quats_F)
                rotmat = ME.SparseTensor(rotmat_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                betas_F = params_F[:, 10:11].clamp(min=0.0, max=10.0)
                betas = ME.SparseTensor(betas_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)

                prob_surf = sort_sparse_tensor(prob_surf, params)
                m = prob_surf.F.squeeze()>0.90
                surf_c_list.append(prob_surf.C[:, 1:][m])
                surf_m_list.append(means.F[m])
                surf_v_list.append(scales.F[m])
                surf_q_list.append(rotmat.F[m])
                surf_bt_list.append(betas.F[m])
                surf_l_list.append(torch.ones_like(surf_c_list[-1][:, :1], dtype=torch.float32)*i)

                prob_surf_top = upsample_to_top(prob_surf)
                idx_top = upsample_to_top(idxs)
                mean_top = upsample_to_top(means)
                # relative coordinate (center is mean) within each surfel
                c_top = mean_top.C[:, 1:] % stride - mean_top.F
                rotmat_top = upsample_to_top(rotmat)
                rotmat_top_F = rotmat_top.F.reshape(rotmat_top.F.size()[0], 3, 3)
                scales_top = upsample_to_top(scales)
                scales_top_F = scales_top.F
                betas_top = upsample_to_top(betas)
                betas_top_F = betas_top.F
                p_F = gaussian_pdf_aniso(c_top, scales_top_F, rotmat_top_F, betas_top_F)
                # to balance positive and negative labels, only top2k voxels are selected for loss calculation. Or the large number of negative voxels will lead to suboptimality (estimate all voxels as negative). 
                if p_F.size()[0]>num_points*2:
                    _, mask = torch.topk(p_F, k=num_points*2, dim=0, largest=True, sorted=False)
                    mask = torch.zeros(p_F.size()[0], dtype=torch.bool, device=p_F.device).scatter_(0, mask, 1)
                else:
                    mask = torch.ones(p_F.size()[0], dtype=torch.bool, device=p_F.device)
                c = mean_top.C[mask]
                c_gt = f2.C
                # similar to SparsePCGC, we need to force the ground truth points into loss calculation. This is only for loss, not for real decoding.
                if keep_adaptive:
                    c = torch.unique(torch.cat([c, c_gt], dim=0), dim=0)
                p = ME.SparseTensor(p_F.unsqueeze(-1), mean_top.C)
                p = p.features_at_coordinates(c.to(torch.float32))
                idx = idx_top.features_at_coordinates(c.to(torch.float32))

                topkplus1 = topkplus1_coordinates(
                    p_F, mean_top.C, int(2**i), surfel_input
                )
                if m.any():
                    topkp1_list.append(topkplus1[m])
                    topkp1_idx_list.append(idxs.F[m])

                c_list.append(c)
                prob_surf_c = prob_surf.features_at_coordinates(c.to(torch.float32))
                p_list.append(p*prob_surf_c)
                idx_list.append(idx)
                l_list = torch.cat([l_list, torch.ones_like(p)*i], dim=0)
                # distortion calculation. Note that only if ground truth points is added (keep_adaptive) can we calculate distortion. Otherwise, this value is meaningless.
                if keep_adaptive:
                    c_isin = f2.features_at_coordinates(c.to(torch.float32))>0.5
                    bcefunc = nn.BCELoss(reduction='none')
                    p = p.clamp(min=1e-8, max=1-1e-8)
                    print(p.size(), c_isin.size())
                    bce_layer = bcefunc(p, c_isin.to(torch.float32))
                    bce_layer = ME.SparseTensor(bce_layer, coordinates=c, tensor_stride=1)
                    sumPooling = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=3)
                    while bce_layer.tensor_stride[0]!=ys_rec.tensor_stride[0]:
                        bce_layer = sumPooling(bce_layer)
                    bce_layer = sort_sparse_tensor(bce_layer, prob_surf)
                    bce_layer = bce_layer.F*prob_surf.F
                    factor = 1-0.275*(i-1)
                    bce_list[i] = bce_layer.sum()/ num_points*factor
                if i<=self.octree_end:
                    break

        surf_c, surf_m, surf_v, surf_q, surf_bt, surf_l = torch.cat(surf_c_list, dim=0), torch.cat(surf_m_list, dim=0), torch.cat(surf_v_list, dim=0), torch.cat(surf_q_list, dim=0), torch.cat(surf_bt_list, dim=0), torch.cat(surf_l_list, dim=0)
        
        prob_sum = [str(x)[:4] for x in prob_sum]
        print('quant sum', quant_latent_sum, 'prob sum', prob_surf_sum, 'prob sum 1', [str(x)[:4] for x in prob_sum])
        bce = bce_list.sum()
        bits = sum(latent_bits_list) + sum(octree_bits_list)
        bpp = bits / num_points
        c_list = torch.cat(c_list, dim=0)
        p_list = torch.cat(p_list, dim=0)
        idx_list = torch.cat(idx_list, dim=0)
        topkp1_list = torch.cat(topkp1_list, dim=0)
        topkp1_idx_list = torch.cat(topkp1_idx_list, dim=0)
        returnDict = {'bce': bce, 'bpp': bpp, 'coordinates': c_list, 'occupancies': p_list, 'surfel_indexes': idx_list, 'layers': l_list, 'surf_c': surf_c, 'surf_m': surf_m, 'surf_v': surf_v, 'surf_q': surf_q, 'surf_bt': surf_bt, 'surf_l': surf_l, 'topkplus1': topkp1_list, 'topkplus1_idx': topkp1_idx_list}
        return returnDict


    def encode(self, f1, f2, device, epoch=99999, show_motion=False, showdir='./show', frame_idx=999, selectionMode='gumbel', keep_adaptive=False, rho=1.0, return_bce=True, return_recon=False, lamb=10, step=10000, frame_name=None):
        ''' initialization '''
        # make sure f2 has no color input
        f2 = ME.SparseTensor(torch.ones_like(f2.F[:, :1]), coordinates=f2.C, tensor_stride=f2.tensor_stride[0], device=f2.device)
        num_points = f2.C.size()[0]
        ys = [0 for i in range(self.layers)]
        ys[0] = f2
        latents = [0 for i in range(self.layers)]
        quant_latents = [0 for i in range(self.layers)]
        octree_bits_list = torch.zeros(self.layers, dtype=torch.float32, device=device)
        latent_bits_list = torch.zeros(self.layers, dtype=torch.float32, device=device)
        bce_list = torch.zeros(self.layers, dtype=torch.float32, device=device)
        l_list = torch.tensor([], dtype=torch.long, device=f2.device)
        c_list, p_list, idx_list, surf_c_list, surf_l_list, surf_m_list, surf_v_list, surf_q_list, surf_bt_list = [], [], [], [], [], [], [], [], []
        quant_latent_sum, prob_sum, prob_surf_sum = [], [], []

        ######## encoder ########
        for i in range(1, self.layers):
            ys[i] = self.enc[i-1](ys[i-1])

        downsample = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        # for code consistency, we downsample the last layer.
        ysd = downsample(ys[-1])
        # prob: probability of each node being octree node; _ste: straight-through estimator for prob (see P-SOPA); _up: upsampled version of prob, the upsample here is fixed-coordinate because we will losslessly code the octants.
        prob = ME.SparseTensor(torch.ones(ysd.C.size()[0], 1, dtype=torch.float32, device=ysd.device), coordinate_manager=ysd.coordinate_manager, coordinate_map_key=ysd.coordinate_map_key)
        prob_up = ME.SparseTensor(torch.ones(ys[-1].C.size()[0], 1, dtype=torch.float32, device=ys[-1].device), coordinate_manager=ys[-1].coordinate_manager, coordinate_map_key=ys[-1].coordinate_map_key, device=ys[-1].device)
        prob_ste = ME.SparseTensor(torch.ones_like(prob.F, device=prob.device), coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key, device=prob.device)
        prob_ste_up = ME.SparseTensor(torch.ones(ys[-1].C.size()[0], 1, dtype=torch.float32, device=ys[-1].device), coordinate_manager=ys[-1].coordinate_manager, coordinate_map_key=ys[-1].coordinate_map_key, device=ys[-1].device)

        gsParams = None
        ys_pred = ME.SparseTensor(torch.zeros_like(ys[-1].F), coordinate_manager=ys[-1].coordinate_manager, coordinate_map_key=ys[-1].coordinate_map_key, device=ys[-1].device)
        topkplus1_list, topkplus1_idx_list = [], []
        surf_counter = 0

        for i in range(self.layers-1, 0, -1):

            ############ latent compression ############
            if i>=self.feature_to:
                latents[i] = self.contextDetach[i](ys[i], ys_pred)
                if i==self.layers-1:
                    # last layer has no context because it's the first to be decoded
                    ctx = None
                else:
                    ctx = [quant_latents[i+1], ys_pred]
                compressionOutput = self.entropy[i](latents[i], training=False, gaussion=True, require_quant=True, ctx=ctx, bar=self.bar)
                quant_latent_sum.append(compressionOutput['x_ada'].F.round().abs().sum().item())
                bs = self.entropy[i].encode(latents[i], training=False, gaussion=True, ctx=ctx, bar=self.bar)
                template = ME.SparseTensor(torch.zeros_like(latents[i].F), coordinate_manager=latents[i].coordinate_manager, coordinate_map_key=latents[i].coordinate_map_key, device=latents[i].device)
                latent_bits_list[i] = len(bs)*8
                quant_latents[i] = self.entropy[i].decode(template, bs, gaussion=True, ctx=ctx, bar=self.bar)
                quant_latents[i] = ME.SparseTensor(quant_latents[i].F, coordinate_manager=latents[i].coordinate_manager, coordinate_map_key=latents[i].coordinate_map_key, device=latents[i].device)
                ys_rec = self.contextRestore[i](quant_latents[i], ys_pred)
            else:
                ys_rec = ys_pred

            ##### Tree Decision #####
            upsampleNonExpand = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3, expand_coordinates=True)
            prob = upsampleNonExpand(prob, ys_pred.C)
            prob = ME.SparseTensor(
                prob.F,
                coordinate_manager=ys_pred.coordinate_manager,
                coordinate_map_key=ys_pred.coordinate_map_key,
                device=ys_pred.device,
            )
            decision_input_F = ys_rec.F
            decision_input = ME.SparseTensor(torch.cat([decision_input_F, torch.zeros_like(decision_input_F)], dim=-1), coordinate_manager=ys_rec.coordinate_manager, coordinate_map_key=ys_rec.coordinate_map_key, device=ys_rec.device)
            if i<=self.octree_end:
                # for any layer finer than octree_end, nodes will be forced to be surfel nodes. This option is for training speedup because finer layers' lossless compression is costly and slow.
                prob_surf = prob
                prob = ME.SparseTensor(torch.zeros_like(prob.F), coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key, device=prob.device)
            elif i<=self.surfel_start:
                # for any layer coarser than surfel_start, nodes will be forced to be octree nodes. This is because the lower layer's octree lossless compression costs not too much bit rates.
                prob_condition = sort_sparse_tensor(self.decision[i](decision_input), prob).F
                if self.selection_mode == 'softmax':
                    prob_condition = torch.softmax(prob_condition, dim=1)[:, :1]
                elif self.selection_mode == 'gumbel':
                    tau = max(0.1, math.exp(-0.00035*step)*1.5)
                    prob_condition = F.gumbel_softmax(prob_condition, tau=tau, hard=False, dim=-1)[:, :1]
                if self.selection_mode == 'fixed':
                    # if selectionMode is fixed, every layer will have a fixed probability of being surfel nodes (note that the decision is still stocastic, but the probability fixed).
                    prob_surf = ME.SparseTensor(torch.ones_like(prob_condition)*(1/(self.surfel_start-self.octree_end+1)), coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key)
                    prob = prob-prob_surf
                else:
                    prob_surf = prob*(1-prob_condition)
                    prob = prob*prob_condition
            else:
                # if i>self.surfel_start, it means all nodes are octree nodes.
                prob_surf = ME.SparseTensor(torch.zeros_like(prob.F), coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key, device=prob.device)

            # P-SOPA STE
            prob_ste_F = torch.rand_like(prob.F, device=prob.device)<=prob.F
            prob_ste_F = prob_ste_F.float()
            prob_ste_F = (prob_ste_F-prob.F).detach()+prob.F
            prob_ste = ME.SparseTensor(prob_ste_F, coordinate_manager=prob.coordinate_manager, coordinate_map_key=prob.coordinate_map_key, device=prob.device)

            if i<=self.surfel_start:
                prob_sum.append((prob.F.sum()/ys[i].size()[0]).item())
                prob_surf_sum.append((prob_surf.F.sum()/ys[i].size()[0]).item())
                ys_rec = sort_sparse_tensor(ys_rec, prob)
                prob_surf = sort_sparse_tensor(prob_surf, ys_rec)
                

            ######### octree reconstruction (for next layer) ##########
            octree_mask = prob.F.squeeze(1)>0.90
            surfel_mask = prob_surf.F.squeeze(1)>0.90
            if i>self.octree_end:
                # when real coding, nodes with prob<0.9 will be pruned.
                if (octree_mask>0.1).any():
                    ys_rec_masked = ME.SparseTensor(ys_rec.F[octree_mask], coordinates=ys_rec.C[octree_mask], tensor_stride=ys_rec.tensor_stride[0])
                    upsample = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3, expand_coordinates=True)
                    ys_rec_masked_up = upsample(ys_rec_masked)
                    ys_occ = ME.SparseTensor(torch.ones_like(ys_rec_masked_up.F[:, :1]), coordinates=ys_rec_masked_up.C, tensor_stride=ys_rec_masked_up.tensor_stride[0])
                    ys_mask = ys_occ.features_at_coordinates(ys[i-1].C.to(torch.float32)).squeeze(1)>0.9
                    ys_ = ME.SparseTensor(ys[i-1].F[ys_mask], coordinates=ys[i-1].C[ys_mask], tensor_stride=ys[i-1].tensor_stride[0])
                    ys_ = sort_sparse_tensor(ys_)
                    # TBD: change this into real entropy coding.
                    octree_out_set = self.dec[i](ys_rec_masked, ys_, training=self.training)
                    ys_pred = octree_out_set['out']
                    ys_pred = sort_sparse_tensor(ys_pred)
                    prob_up, prob_ste_up = octree_out_set['prob_up'], octree_out_set['prob_ste_up']
                    out_cls, target = octree_out_set['out_cls_list'], octree_out_set['target_list']
                    out_cls = ME.SparseTensor(torch.cat([out_cls[i].F for i in range(len(out_cls))], dim=0), coordinates=torch.cat([out_cls[i].C for i in range(len(out_cls))], dim=0), tensor_stride=ys[i-1].tensor_stride[0])
                    target = torch.cat(target, dim=0)
                    bcefunc = nn.BCEWithLogitsLoss(reduction='none')
                    
                    octree_bits = bcefunc(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))/math.log(2.0)
                    octree_bits_list[i] = octree_bits.sum()

            ######## Surfel Reconstruction ########
            if i<=self.surfel_start:
                if (surfel_mask>0.1).any():
                    ys_rec_surf_masked = ME.SparseTensor(ys_rec.F[surfel_mask], coordinates=ys_rec.C[surfel_mask], tensor_stride=ys_rec.tensor_stride[0])
                    surfel_input = ME.SparseTensor(torch.cat([ys_rec_surf_masked.F, torch.zeros_like(ys_rec_surf_masked.F)], dim=-1), coordinates=ys_rec_surf_masked.C, tensor_stride=ys_rec_surf_masked.tensor_stride[0])
                    params = self.GSdec[i](surfel_input)
                    stride = params.tensor_stride[0]
                    params_F = params.F
                    idxs = ME.SparseTensor(torch.arange(params.C.size()[0], device=params.device, dtype=torch.float32).unsqueeze(1)+surf_counter,  coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                    surf_counter += params.C.size()[0]
                    means = ME.SparseTensor(params_F[:, :3], coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                    scales_F = torch.exp(params_F[:, 3:6].clamp(min=-8, max=8))
                    scales = ME.SparseTensor(scales_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                    quats_F = params_F[:, 6:10]
                    quats_F = quats_F / (quats_F.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)
                    quats = ME.SparseTensor(quats_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                    rotmat_F = quat2rotmat(quats_F)
                    rotmat = ME.SparseTensor(rotmat_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)
                    betas_F = params_F[:, 10:11].clamp(min=0.0, max=10.0)
                    betas = ME.SparseTensor(betas_F, coordinate_map_key=params.coordinate_map_key, coordinate_manager=params.coordinate_manager, device=params.device)

                    surf_c_list.append(prob_surf.C[:, 1:])
                    surf_m_list.append(means.F)
                    surf_v_list.append(scales.F)
                    surf_q_list.append(rotmat.F)
                    surf_bt_list.append(betas.F)
                    surf_l_list.append(torch.ones_like(surf_c_list[-1][:, :1], dtype=torch.float32)*i)

                    idx_top = upsample_to_top(idxs)
                    mean_top = upsample_to_top(means)
                    # relative coordinate (center is mean) within each surfel
                    c_top = mean_top.C[:, 1:] % stride - mean_top.F
                    rotmat_top = upsample_to_top(rotmat)
                    rotmat_top_F = rotmat_top.F.reshape(rotmat_top.F.size()[0], 3, 3)
                    scales_top = upsample_to_top(scales)
                    scales_top_F = scales_top.F
                    betas_top = upsample_to_top(betas)
                    betas_top_F = betas_top.F
                    p_F = gaussian_pdf_aniso(c_top, scales_top_F, rotmat_top_F, betas_top_F)
                    # to balance positive and negative labels, only top2k voxels are selected for loss calculation. Otherwise the large number of negative voxels will lead to suboptimality (estimate all voxels as negative). 
                    if p_F.size()[0]>num_points*2:
                        _, mask = torch.topk(p_F, k=num_points*2, dim=0, largest=True, sorted=False)
                        mask = torch.zeros(p_F.size()[0], dtype=torch.bool, device=p_F.device).scatter_(0, mask, 1)
                    else:
                        mask = torch.ones(p_F.size()[0], dtype=torch.bool, device=p_F.device)
                    c = mean_top.C[mask]
                    c_gt = f2.C
                    # similar to SparsePCGC, we need to force the ground truth points into loss calculation. This is only for loss, not for real decoding.
                    if keep_adaptive:
                        c = torch.unique(torch.cat([c, c_gt], dim=0), dim=0)
                    p = ME.SparseTensor(p_F.unsqueeze(-1), mean_top.C)
                    p = p.features_at_coordinates(c.to(torch.float32))
                    idx = idx_top.features_at_coordinates(c.to(torch.float32))
                    # topk+1
                    bounding_box_size = int(2**i)
                    topkplus1 = topkplus1_coordinates(
                        p_F, mean_top.C, bounding_box_size, surfel_input
                    )
                    if topkplus1.size(0):
                        topkplus1_list.append(topkplus1)
                        topkplus1_idx_list.append(idxs.F)

                    c_list.append(c)
                    p_list.append(p)
                    idx_list.append(idx)
                    l_list = torch.cat([l_list, torch.ones_like(p)*i], dim=0)
                if i<=self.octree_end:
                    break

        surf_c, surf_m, surf_v, surf_q, surf_bt, surf_l = torch.cat(surf_c_list, dim=0), torch.cat(surf_m_list, dim=0), torch.cat(surf_v_list, dim=0), torch.cat(surf_q_list, dim=0), torch.cat(surf_bt_list, dim=0), torch.cat(surf_l_list, dim=0)
        
        prob_sum = [str(x)[:4] for x in prob_sum]
        print('quant sum', quant_latent_sum, 'prob sum', [str(x)[:4] for x in prob_surf_sum], 'prob sum 1', [str(x)[:4] for x in prob_sum])
        bce = bce_list.sum()
        bits = sum(latent_bits_list) + sum(octree_bits_list)
        bpp = bits / num_points
        c_list = torch.cat(c_list, dim=0)
        p_list = torch.cat(p_list, dim=0)
        idx_list = torch.cat(idx_list, dim=0)
        topkplus1_list = torch.cat(topkplus1_list, dim=0)
        topkplus1_idx_list = torch.cat(topkplus1_idx_list, dim=0)
        returnDict = {'bce': bce, 'bpp': bpp, 'coordinates': c_list, 'occupancies': p_list, 'surfel_indexes': idx_list, 'layers': l_list, 'surf_c': surf_c, 'surf_m': surf_m, 'surf_v': surf_v, 'surf_q': surf_q, 'surf_bt': surf_bt, 'surf_l': surf_l, 'topkplus1': topkplus1_list, 'topkplus1_idx': topkplus1_idx_list}
        return returnDict