from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning import LightningModule

class RTT(LightningModule):
    def __init__(
        self,
        model,
        loss_function,
        onset_weight,
        offset_weight,
        frame_weight,
        learning_rate,
        lr_scheduler,
        weight_decay, 
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_function'])
        self.model = model
        self.loss_function = loss_function
        self.onset_weight = onset_weight
        self.offset_weight = offset_weight
        self.frame_weight = frame_weight
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
    
    def training_step(self):
        return
    
    def validation_step(self):
        return
        