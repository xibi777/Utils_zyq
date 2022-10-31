import torch
from detectron2.projects.point_rend.point_features import point_sample
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast
from mask2former.modeling.matcher import batch_dice_loss, batch_sigmoid_ce_loss_jit

# N个mask proposal与target中的mask匹配
# 其中num_point,self.cost_mask,self.cost_dice为超参数

indices = []
num_queries = mask_pred_result.shape[0]
out_mask = mask_pred_result  # [num_queries, H_pred, W_pred]
# gt masks are already padded when preparing target
tgt_mask = targets[0]["masks"].to(out_mask)

out_mask = out_mask[:, None]
tgt_mask = tgt_mask[:, None]
# all masks share the same set of points for efficient matching!
point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
# get gt labels
tgt_mask = point_sample(
    tgt_mask,
    point_coords.repeat(tgt_mask.shape[0], 1, 1),
    align_corners=False,
).squeeze(1)

out_mask = point_sample(
    out_mask,
    point_coords.repeat(out_mask.shape[0], 1, 1),
    align_corners=False,
).squeeze(1)

with autocast(enabled=False):
    out_mask = out_mask.float()
    tgt_mask = tgt_mask.float()
    # Compute the focal loss between masks
    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

    # Compute the dice loss betwen masks
    batch_dice_loss_jit = torch.jit.script(
        batch_dice_loss
    )
    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

# Final cost matrix
C = (
        self.cost_mask * cost_mask
        # + self.cost_class * cost_class
        + self.cost_dice * cost_dice
)
C = C.reshape(num_queries, -1).cpu()

indices.append(linear_sum_assignment(C))
indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
for i, j in indices]

