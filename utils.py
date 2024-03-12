from sklearn.metrics import roc_auc_score
import torch,math
def compute_AUCs(gt, pred, N_CLASSES=14):
    """计算AUC值.

    参数:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    返回:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, last_epoch=-1, min_lr=0):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup阶段的学习率计算
            lr_scale = self.last_epoch / float(self.warmup_steps)
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段的学习率计算
            progress = self.last_epoch - self.warmup_steps
            max_progress = self.total_steps - self.warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress / max_progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs]
