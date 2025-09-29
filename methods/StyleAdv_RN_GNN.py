import torch
import torch.nn as nn
import numpy as np
import random

from methods.gnn import GNN_nl
from methods import backbone_multiblock
from methods.tool_func import *
from methods.meta_template_StyleAdv_RN_GNN import MetaTemplate


class StyleAdvGNN(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support, tf_path=None,
               enable_factor_decomposition=False,
               enable_mutual_info_loss=False,
               feature_dims=[64, 128, 256],
               enable_policy=False
              ):
    super(StyleAdvGNN, self).__init__(model_func, n_way, n_support, tf_path=tf_path,
                                      enable_factor_decomposition=enable_factor_decomposition,
                                      enable_mutual_info_loss=enable_mutual_info_loss,
                                      feature_dims=feature_dims,
                                      enable_policy=enable_policy)

    if enable_factor_decomposition:
        self.block_decomposers = nn.ModuleDict()
        self.block_reconstructors = nn.ModuleDict()
        self.block_mi_estimators = nn.ModuleDict()

        for i, dim in enumerate(feature_dims):
            block_name = f'block{i + 1}'

            # Identity/Style decomposer for each block
            self.block_decomposers[f'{block_name}_identity'] = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.ReLU(),
                nn.Linear(dim // 2, dim // 4)
            )
            self.block_decomposers[f'{block_name}_style'] = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.ReLU(),
                nn.Linear(dim // 2, dim // 4)
            )

            # Factor reconstructor for each block
            self.block_reconstructors[block_name] = nn.Sequential(
                nn.Linear(dim // 2, dim // 2), nn.ReLU(),
                nn.Linear(dim // 2, dim)
            )

            # Mutual information estimator for each block
            if enable_mutual_info_loss:
                self.block_mi_estimators[block_name] = nn.Sequential(
                    nn.Linear(dim // 2, dim // 4), nn.ReLU(),
                    nn.Linear(dim // 4, 1)
                )

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)

    # for global classifier
    self.method = 'GnnNet'
    self.classifier = nn.Linear(self.feature.final_feat_dim, 64)

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.classifier.cuda()
    self.support_label = self.support_label.cuda()

    if self.enable_factor_decomposition:
      self.block_decomposers.cuda()
      self.block_reconstructors.cuda()
      if self.enable_mutual_info_loss:
          self.block_mi_estimators.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores
        
  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

  def decompose_block_representation(self, block_features, block_name):
    if not self.enable_factor_decomposition:
        return None, None, block_features, 0, 0

    # Global average pooling for spatial features
    if len(block_features.shape) == 4:  # [B, C, H, W]
        pooled_features = F.adaptive_avg_pool2d(block_features, 1).squeeze(-1).squeeze(-1)
    else:
        pooled_features = block_features

    # Decompose into identity and style factors
    identity_factor = self.block_decomposers[f'{block_name}_identity'](pooled_features)
    style_factor = self.block_decomposers[f'{block_name}_style'](pooled_features)

    # Reconstruct original representation
    factors_concat = torch.cat([identity_factor, style_factor], dim=1)
    reconstructed_features = self.block_reconstructors[block_name](factors_concat)

    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed_features, pooled_features)

    # Mutual information loss
    mi_loss = self.compute_block_mutual_info_loss(identity_factor, style_factor, block_name)
  
    return identity_factor, style_factor, reconstructed_features, recon_loss, mi_loss

  def compute_block_mutual_info_loss(self, identity_factor, style_factor, block_name):
    if not self.enable_mutual_info_loss:
        return 0

    batch_size = identity_factor.size(0)

    # Joint distribution
    factors_joint = torch.cat([identity_factor, style_factor], dim=1)
    joint_scores = self.block_mi_estimators[block_name](factors_joint)

    # Marginal distribution (shuffle factors)
    identity_shuffled = identity_factor[torch.randperm(batch_size)]
    factors_marginal = torch.cat([identity_shuffled, style_factor], dim=1)
    marginal_scores = self.block_mi_estimators[block_name](factors_marginal)

    # Donsker-Varadhan estimation
    mi_loss = torch.mean(joint_scores) - torch.log(torch.mean(torch.exp(marginal_scores)) + 1e-8)

    return mi_loss

  def factor_aware_style_attack(self, x_ori, y_ori, epsilon_list):
    """Factor-aware adversarial attack"""
    x_ori = x_ori.cuda()
    y_ori = y_ori.cuda()
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
    y_ori = y_ori.view(x_size[0] * x_size[1])

    # Store adversarial factors for each block
    adv_factors = {}
    total_recon_loss = 0
    total_mi_loss = 0

    blocklist = 'block123'

    # Block 1
    if ('1' in blocklist and epsilon_list[0] != 0):
        x_ori_block1 = self.feature.forward_block1(x_ori)

        # Factor decomposition
        identity_factor1, style_factor1, recon_feat1, recon_loss1, mi_loss1 = \
            self.decompose_block_representation(x_ori_block1, 'block1')

        total_recon_loss += recon_loss1
        total_mi_loss += mi_loss1

        # Attack only style factor, preserve identity
        style_factor1_param = torch.nn.Parameter(style_factor1.clone())
        style_factor1_param.requires_grad_()

        # Reconstruct with attacked style for forward pass
        factors_concat = torch.cat([identity_factor1.detach(), style_factor1_param], dim=1)
        attacked_pooled_feat = self.block_reconstructors['block1'](factors_concat)
        if len(x_ori_block1.shape) == 4:
            B, C, H, W = x_ori_block1.shape
            attacked_feat = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        else:
            attacked_feat = attacked_pooled_feat

        # Forward pass with attacked features
        x_ori_block2 = self.feature.forward_block2(attacked_feat)
        x_ori_block3 = self.feature.forward_block3(x_ori_block2)
        x_ori_block4 = self.feature.forward_block4(x_ori_block3)
        x_ori_fea = self.feature.forward_rest(x_ori_block4)
        x_ori_output = self.classifier.forward(x_ori_fea)

        ori_loss = self.loss_fn(x_ori_output, y_ori)

        self.feature.zero_grad()
        self.classifier.zero_grad()
        ori_loss.backward()

        # FGSM attack on style factor
        epsilon = epsilon_list[torch.randint(0, len(epsilon_list), (1,))[0]]
        adv_style_factor1 = style_factor1 + epsilon * torch.sign(style_factor1_param.grad)

        adv_factors['block1'] = {
            'identity': identity_factor1.detach(),
            'style': adv_style_factor1.detach()
        }

    # Block 2
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if ('2' in blocklist and epsilon_list[1] != 0):
        x_ori_block1 = self.feature.forward_block1(x_ori)
        factors_concat = torch.cat([
            adv_factors['block1']['identity'],
            adv_factors['block1']['style']
        ], dim=1)
        attacked_pooled_feat = self.block_reconstructors['block1'](factors_concat)
        if len(x_ori_block1.shape) == 4:
            B, C, H, W = x_ori_block1.shape
            x_adv_block1 = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        else:
            x_adv_block1 = attacked_pooled_feat

        x_ori_block2 = self.feature.forward_block2(x_adv_block1)

        # Factor decomposition for block2
        identity_factor2, style_factor2, recon_feat2, recon_loss2, mi_loss2 = \
            self.decompose_block_representation(x_ori_block2, 'block2')

        total_recon_loss += recon_loss2
        total_mi_loss += mi_loss2

        # Attack only style factor, preserve identity
        style_factor2_param = torch.nn.Parameter(style_factor2.clone())
        style_factor2_param.requires_grad_()

        # Reconstruct with attacked style for forward pass
        factors_concat = torch.cat([identity_factor2.detach(), style_factor2_param], dim=1)
        attacked_pooled_feat = self.block_reconstructors['block2'](factors_concat)
        if len(x_ori_block2.shape) == 4:
            B, C, H, W = x_ori_block2.shape
            attacked_feat = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        else:
            attacked_feat = attacked_pooled_feat

        # Forward pass with attacked features
        x_ori_block3 = self.feature.forward_block3(attacked_feat)
        x_ori_block4 = self.feature.forward_block4(x_ori_block3)
        x_ori_fea = self.feature.forward_rest(x_ori_block4)
        x_ori_output = self.classifier.forward(x_ori_fea)

        ori_loss = self.loss_fn(x_ori_output, y_ori)

        self.feature.zero_grad()
        self.classifier.zero_grad()
        ori_loss.backward()

        # FGSM attack on style factor
        epsilon = epsilon_list[torch.randint(0, len(epsilon_list), (1,))[0]]
        adv_style_factor2 = style_factor2 + epsilon * torch.sign(style_factor2_param.grad)

        adv_factors['block2'] = {
            'identity': identity_factor2.detach(),
            'style': adv_style_factor2.detach()
        }

    if ('3' in blocklist and epsilon_list[2] != 0):
        x_ori_block1 = self.feature.forward_block1(x_ori)
        factors_concat = torch.cat([
            adv_factors['block1']['identity'],
            adv_factors['block1']['style']
        ], dim=1)
        attacked_pooled_feat = self.block_reconstructors['block1'](factors_concat)
        if len(x_ori_block1.shape) == 4:
            B, C, H, W = x_ori_block1.shape
            x_adv_block1 = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        else:
            x_adv_block1 = attacked_pooled_feat

        x_ori_block2 = self.feature.forward_block2(x_adv_block1)
        factors_concat = torch.cat([
            adv_factors['block2']['identity'],
            adv_factors['block2']['style']
        ], dim=1)
        attacked_pooled_feat = self.block_reconstructors['block2'](factors_concat)
        if len(x_ori_block2.shape) == 4:
            B, C, H, W = x_ori_block2.shape
            x_adv_block2 = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        else:
            x_adv_block2 = attacked_pooled_feat

        x_ori_block3 = self.feature.forward_block3(x_adv_block2)

        # Factor decomposition for block3
        identity_factor3, style_factor3, recon_feat3, recon_loss3, mi_loss3 = \
            self.decompose_block_representation(x_ori_block3, 'block3')

        total_recon_loss += recon_loss3
        total_mi_loss += mi_loss3

        # Attack only style factor, preserve identity
        style_factor3_param = torch.nn.Parameter(style_factor3.clone())
        style_factor3_param.requires_grad_()

        # Reconstruct with attacked style for forward pass
        factors_concat = torch.cat([identity_factor3.detach(), style_factor3_param], dim=1)
        attacked_pooled_feat = self.block_reconstructors['block3'](factors_concat)
        if len(x_ori_block3.shape) == 4:
            B, C, H, W = x_ori_block3.shape
            attacked_feat = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        else:
            attacked_feat = attacked_pooled_feat

        # Forward pass with attacked features
        x_ori_block4 = self.feature.forward_block4(attacked_feat)
        x_ori_fea = self.feature.forward_rest(x_ori_block4)
        x_ori_output = self.classifier.forward(x_ori_fea)

        ori_loss = self.loss_fn(x_ori_output, y_ori)

        self.feature.zero_grad()
        self.classifier.zero_grad()
        ori_loss.backward()

        # FGSM attack on style factor
        epsilon = epsilon_list[torch.randint(0, len(epsilon_list), (1,))[0]]
        adv_style_factor3 = style_factor3 + epsilon * torch.sign(style_factor3_param.grad)

        adv_factors['block3'] = {
            'identity': identity_factor3.detach(),
            'style': adv_style_factor3.detach()
        }

    return adv_factors, total_recon_loss, total_mi_loss

  def set_forward_loss_with_factors(self, x_ori, global_y, epsilon_list):
    """Factor-aware forward pass"""
    ##################################################################
    # 0. first cp x_adv from x_ori
    x_adv = x_ori

    ##################################################################
    # 1. styleAdv
    self.set_statues_of_modules('eval')

    # Get adversarial factors and losses
    adv_factors, total_recon_loss, total_mi_loss = \
        self.factor_aware_style_attack(x_ori, global_y, epsilon_list)

    self.feature.zero_grad()
    self.fc.zero_grad()
    self.classifier.zero_grad()
    self.gnn.zero_grad()

    #################################################################
    self.set_statues_of_modules('train')

    # define y_query for FSL
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.cuda()

    # Original forward pass
    x_ori = x_ori.cuda()
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
    global_y = global_y.view(x_size[0] * x_size[1]).cuda()
    x_ori_block1 = self.feature.forward_block1(x_ori)
    x_ori_block2 = self.feature.forward_block2(x_ori_block1)
    x_ori_block3 = self.feature.forward_block3(x_ori_block2)
    x_ori_block4 = self.feature.forward_block4(x_ori_block3)
    x_ori_fea = self.feature.forward_rest(x_ori_block4)

    scores_cls_ori = self.classifier.forward(x_ori_fea)
    loss_cls_ori = self.loss_fn(scores_cls_ori, global_y)

    # (2) Few-shot(GNN)
    x_ori_z = self.fc(x_ori_fea) #  (d→z)
    x_ori_z = x_ori_z.view(self.n_way, -1, x_ori_z.size(1))
    x_ori_z_stack = [
        torch.cat([x_ori_z[:, :self.n_support], x_ori_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(
            1, -1, x_ori_z.size(2)) for i in range(self.n_query)]
    assert (x_ori_z_stack[0].size(1) == self.n_way * (self.n_support + 1))
    scores_fsl_ori = self.forward_gnn(x_ori_z_stack)
    loss_fsl_ori = self.loss_fn(scores_fsl_ori, y_query)

    # Adversarial forward pass with factor application
    x_adv = x_adv.cuda()
    x_adv = x_adv.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])

    # Apply adversarial factors...
    x_adv_block1 = self.feature.forward_block1(x_adv)
    factors_concat = torch.cat([
        adv_factors['block1']['identity'],
        adv_factors['block1']['style']
    ], dim=1)
    attacked_pooled_feat = self.block_reconstructors['block1'](factors_concat)
    if len(x_adv_block1.shape) == 4:
        B, C, H, W = x_adv_block1.shape
        x_adv_block1 = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
    else:
        x_adv_block1 = attacked_pooled_feat

    x_adv_block2 = self.feature.forward_block2(x_adv_block1)
    factors_concat = torch.cat([
        adv_factors['block2']['identity'],
        adv_factors['block2']['style']
    ], dim=1)
    attacked_pooled_feat = self.block_reconstructors['block2'](factors_concat)
    if len(x_adv_block2.shape) == 4:
        B, C, H, W = x_adv_block2.shape
        x_adv_block2 = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
    else:
        x_adv_block2 = attacked_pooled_feat

    x_adv_block3 = self.feature.forward_block3(x_adv_block2)
    factors_concat = torch.cat([
        adv_factors['block3']['identity'],
        adv_factors['block3']['style']
    ], dim=1)
    attacked_pooled_feat = self.block_reconstructors['block3'](factors_concat)
    if len(x_adv_block3.shape) == 4:
        B, C, H, W = x_adv_block3.shape
        x_adv_block3 = attacked_pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
    else:
        x_adv_block3 = attacked_pooled_feat

    x_adv_block4 = self.feature.forward_block4(x_adv_block3)
    x_adv_fea = self.feature.forward_rest(x_adv_block4)
    scores_cls_adv = self.classifier.forward(x_adv_fea)
    loss_cls_adv = self.loss_fn(scores_cls_adv, global_y)

    # Adversarial FSL
    x_adv_z = self.fc(x_adv_fea)
    x_adv_z = x_adv_z.view(self.n_way, -1, x_adv_z.size(1))
    x_adv_z_stack = [
        torch.cat([x_adv_z[:, :self.n_support], x_adv_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(
            1, -1, x_adv_z.size(2)) for i in range(self.n_query)]
    assert (x_adv_z_stack[0].size(1) == self.n_way * (self.n_support + 1))
    scores_fsl_adv = self.forward_gnn(x_adv_z_stack)
    loss_fsl_adv = self.loss_fn(scores_fsl_adv, y_query)

    return (scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori,
            scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv,
            total_recon_loss, total_mi_loss)
      
  def adversarial_attack_Incre(self, x_ori, y_ori, epsilon_list):
    x_ori = x_ori.cuda()
    y_ori = y_ori.cuda()
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    y_ori = y_ori.view(x_size[0]*x_size[1])

    # if not adv, set defalut = 'None'
    adv_style_mean_block1, adv_style_std_block1 = 'None', 'None'
    adv_style_mean_block2, adv_style_std_block2 = 'None', 'None'
    adv_style_mean_block3, adv_style_std_block3 = 'None', 'None'

    # forward and set the grad = True
    blocklist = 'block123'
    
    if('1' in blocklist and epsilon_list[0] != 0 ):
      # forward block1
      x_ori_block1 = self.feature.forward_block1(x_ori)
      feat_size_block1 = x_ori_block1.size()
      ori_style_mean_block1, ori_style_std_block1 = calc_mean_std(x_ori_block1)
      # set them as learnable parameters
      ori_style_mean_block1  = torch.nn.Parameter(ori_style_mean_block1)
      ori_style_std_block1 = torch.nn.Parameter(ori_style_std_block1)
      ori_style_mean_block1.requires_grad_()
      ori_style_std_block1.requires_grad_()
      # contain ori_style_mean_block1 in the graph 
      x_normalized_block1 = (x_ori_block1 - ori_style_mean_block1.detach().expand(feat_size_block1)) / ori_style_std_block1.detach().expand(feat_size_block1)
      x_ori_block1 = x_normalized_block1 * ori_style_std_block1.expand(feat_size_block1) + ori_style_mean_block1.expand(feat_size_block1)
      
      # pass the rest model
      x_ori_block2 = self.feature.forward_block2(x_ori_block1)
      x_ori_block3 = self.feature.forward_block3(x_ori_block2)
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
    
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]

      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
   
      # backward loss
      ori_loss.backward()

      # collect datagrad
      grad_ori_style_mean_block1 = ori_style_mean_block1.grad.detach()
      grad_ori_style_std_block1 = ori_style_std_block1.grad.detach()
    
      # fgsm style attack
      index = torch.randint(0, len(epsilon_list), (1, ))[0]
      epsilon = epsilon_list[index]

      adv_style_mean_block1 = fgsm_attack(ori_style_mean_block1, epsilon, grad_ori_style_mean_block1)
      adv_style_std_block1 = fgsm_attack(ori_style_std_block1, epsilon, grad_ori_style_std_block1)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('2' in blocklist and epsilon_list[1] != 0):
      # forward block1
      x_ori_block1 = self.feature.forward_block1(x_ori)
      # update adv_block1
      x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
      # forward block2
      x_ori_block2 = self.feature.forward_block2(x_adv_block1) 
      # calculate mean and std
      feat_size_block2 = x_ori_block2.size()
      ori_style_mean_block2, ori_style_std_block2 = calc_mean_std(x_ori_block2)
      # set them as learnable parameters
      ori_style_mean_block2  = torch.nn.Parameter(ori_style_mean_block2)
      ori_style_std_block2 = torch.nn.Parameter(ori_style_std_block2)
      ori_style_mean_block2.requires_grad_()
      ori_style_std_block2.requires_grad_()
      # contain ori_style_mean_block1 in the graph 
      x_normalized_block2 = (x_ori_block2 - ori_style_mean_block2.detach().expand(feat_size_block2)) / ori_style_std_block2.detach().expand(feat_size_block2)
      x_ori_block2 = x_normalized_block2 * ori_style_std_block2.expand(feat_size_block2) + ori_style_mean_block2.expand(feat_size_block2)
      # pass the rest model
      x_ori_block3 = self.feature.forward_block3(x_ori_block2)
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
      # backward loss
      ori_loss.backward()
      # collect datagrad
      grad_ori_style_mean_block2 = ori_style_mean_block2.grad.detach()
      grad_ori_style_std_block2 = ori_style_std_block2.grad.detach()
      # fgsm style attack
      index = torch.randint(0, len(epsilon_list), (1, ))[0]
      epsilon = epsilon_list[index]
      adv_style_mean_block2 = fgsm_attack(ori_style_mean_block2, epsilon, grad_ori_style_mean_block2)
      adv_style_std_block2 = fgsm_attack(ori_style_std_block2, epsilon, grad_ori_style_std_block2)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('3' in blocklist and epsilon_list[2] != 0):
      # forward block1, block2, block3
      x_ori_block1 = self.feature.forward_block1(x_ori)
      x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
      x_ori_block2 = self.feature.forward_block2(x_adv_block1)
      x_adv_block2 = changeNewAdvStyle(x_ori_block2, adv_style_mean_block2, adv_style_std_block2, p_thred=0)
      x_ori_block3 = self.feature.forward_block3(x_adv_block2)
      # calculate mean and std
      feat_size_block3 = x_ori_block3.size()
      ori_style_mean_block3, ori_style_std_block3 = calc_mean_std(x_ori_block3)
      # set them as learnable parameters
      ori_style_mean_block3  = torch.nn.Parameter(ori_style_mean_block3)
      ori_style_std_block3 = torch.nn.Parameter(ori_style_std_block3)
      ori_style_mean_block3.requires_grad_()
      ori_style_std_block3.requires_grad_()
      # contain ori_style_mean_block3 in the graph 
      x_normalized_block3 = (x_ori_block3 - ori_style_mean_block3.detach().expand(feat_size_block3)) / ori_style_std_block3.detach().expand(feat_size_block3)
      x_ori_block3 = x_normalized_block3 * ori_style_std_block3.expand(feat_size_block3) + ori_style_mean_block3.expand(feat_size_block3)
      # pass the rest model
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
      # backward loss
      ori_loss.backward()
      # collect datagrad
      grad_ori_style_mean_block3 = ori_style_mean_block3.grad.detach()
      grad_ori_style_std_block3 = ori_style_std_block3.grad.detach()
      # fgsm style attack
      index = torch.randint(0, len(epsilon_list), (1, ))[0]
      epsilon = epsilon_list[index]
      adv_style_mean_block3 = fgsm_attack(ori_style_mean_block3, epsilon, grad_ori_style_mean_block3)
      adv_style_std_block3 = fgsm_attack(ori_style_std_block3, epsilon, grad_ori_style_std_block3)

    return adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 
    
  
  def set_statues_of_modules(self, flag):
    if(flag=='eval'):
      self.feature.eval()
      self.fc.eval()
      self.gnn.eval()
      self.classifier.eval()
    elif(flag=='train'):
      self.feature.train()
      self.fc.train()
      self.gnn.train()
      self.classifier.train()
    return 
   

  def set_forward_loss_StyAdv(self, x_ori, global_y, epsilon_list):
    ##################################################################
    # 0. first cp x_adv from x_ori
    x_adv = x_ori

    ##################################################################
    # 1. styleAdv
    self.set_statues_of_modules('eval') 

    adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 = self.adversarial_attack_Incre(x_ori, global_y, epsilon_list)
 
    self.feature.zero_grad()
    self.fc.zero_grad()
    self.classifier.zero_grad()
    self.gnn.zero_grad()

    #################################################################
    # 2. forward and get loss
    self.set_statues_of_modules('train')

    # define y_query for FSL
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()

    # forward x_ori 
    x_ori = x_ori.cuda()
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    global_y = global_y.view(x_size[0]*x_size[1]).cuda()
    x_ori_block1 = self.feature.forward_block1(x_ori)
    x_ori_block2 = self.feature.forward_block2(x_ori_block1)
    x_ori_block3 = self.feature.forward_block3(x_ori_block2)
    x_ori_block4 = self.feature.forward_block4(x_ori_block3)
    x_ori_fea = self.feature.forward_rest(x_ori_block4)

    # ori cls global loss    
    scores_cls_ori = self.classifier.forward(x_ori_fea)
    loss_cls_ori = self.loss_fn(scores_cls_ori, global_y)
    acc_cls_ori = ( scores_cls_ori.max(1, keepdim=True)[1]  == global_y ).type(torch.float).sum().item() / global_y.size()[0]

    # ori FSL scores and losses
    x_ori_z = self.fc(x_ori_fea)
    x_ori_z = x_ori_z.view(self.n_way, -1, x_ori_z.size(1))
    x_ori_z_stack = [torch.cat([x_ori_z[:, :self.n_support], x_ori_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, x_ori_z.size(2)) for i in range(self.n_query)]
    assert(x_ori_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_ori = self.forward_gnn(x_ori_z_stack)
    loss_fsl_ori = self.loss_fn(scores_fsl_ori, y_query)

    # forward x_adv
    x_adv = x_adv.cuda()
    x_adv = x_adv.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    x_adv_block1 = self.feature.forward_block1(x_adv)

    x_adv_block1_newStyle = changeNewAdvStyle(x_adv_block1, adv_style_mean_block1, adv_style_std_block1, p_thred = P_THRED) 
    x_adv_block2 = self.feature.forward_block2(x_adv_block1_newStyle)
    x_adv_block2_newStyle = changeNewAdvStyle(x_adv_block2, adv_style_mean_block2, adv_style_std_block2, p_thred = P_THRED)
    x_adv_block3 = self.feature.forward_block3(x_adv_block2_newStyle)
    x_adv_block3_newStyle = changeNewAdvStyle(x_adv_block3, adv_style_mean_block3, adv_style_std_block3, p_thred = P_THRED)
    x_adv_block4 = self.feature.forward_block4(x_adv_block3_newStyle)
    x_adv_fea = self.feature.forward_rest(x_adv_block4)
   
    # adv cls gloabl loss
    scores_cls_adv = self.classifier.forward(x_adv_fea)
    loss_cls_adv = self.loss_fn(scores_cls_adv, global_y)
    acc_cls_adv = ( scores_cls_adv.max(1, keepdim=True)[1]  == global_y ).type(torch.float).sum().item() / global_y.size()[0]

    # adv FSL scores and losses
    x_adv_z = self.fc(x_adv_fea)
    x_adv_z = x_adv_z.view(self.n_way, -1, x_adv_z.size(1))
    x_adv_z_stack = [torch.cat([x_adv_z[:, :self.n_support], x_adv_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, x_adv_z.size(2)) for i in range(self.n_query)]
    assert(x_adv_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_adv = self.forward_gnn(x_adv_z_stack)
    loss_fsl_adv = self.loss_fn(scores_fsl_adv, y_query)

    #print('scores_fsl_adv:', scores_fsl_adv.mean(), 'loss_fsl_adv:', loss_fsl_adv, 'scores_cls_adv:', scores_cls_adv.mean(), 'loss_cls_adv:', loss_cls_adv)
    return scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv
