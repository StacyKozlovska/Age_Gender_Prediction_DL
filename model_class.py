import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, r2_score
import logging
from typing import Optional, List, Tuple
from torchvision.models import ResNet18_Weights

class MultiTaskModel(pl.LightningModule):
    """
    A multi-task model for gender classification and age regression.
    """

    def __init__(self,
                 unfreeze_layers: Optional[List[str]] = None,
                 feature_extractor: nn.Module = models.resnet18(weights=ResNet18_Weights.DEFAULT),
                 learning_rate: float = 1e-3,
                 initial_epochs: int = 1,
                 dropout_rate: float = 0.5,
                 consistency_loss_weight: float = 0.1,
                 gender_class_weights: Optional[torch.Tensor] = None, 
                 additional_fc: bool = False):
        """
        Initialize the MultiTaskModel.

        Args:
            unfreeze_layers (Optional[List[str]]): List of layer names to 
            unfreeze during fine-tuning.
            feature_extractor (nn.Module): Base feature extractor model.
            learning_rate (float): Initial learning rate.
            initial_epochs (int): Number of epochs to train with frozen base layers.
        """
        super(MultiTaskModel, self).__init__()

        self.lr = learning_rate
        self.initial_epochs = initial_epochs
        self.dropout_rate = dropout_rate
        self.consistency_loss_weight = consistency_loss_weight
        self.gender_class_weights = gender_class_weights

        self.feature_extractor = feature_extractor
        num_features = self.feature_extractor.fc.in_features
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        if additional_fc:
            self.feature_extractor.fc = nn.Sequential(
                nn.Linear(num_features, num_features // 2),
                nn.ReLU(),
                self.dropout,
                nn.Linear(num_features // 2, num_features),
                nn.ReLU()
            )
        
        self.gender_classifier = nn.Linear(num_features, 1)
        self.age_regressor = nn.Linear(num_features, 1)

        self.unfreeze_layers_list = unfreeze_layers

        self.gender_classifier.requires_grad = True
        self.age_regressor.requires_grad = True

        self.train_losses = []
        self.val_losses = []

        self.current_epoch_counter = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gender prediction 
            and age prediction tensors.
        """
        features = self.feature_extractor(x)
        features = F.relu(features)  
        features = self.batch_norm(features)  
        features = self.dropout(features)  
        gender_output = torch.sigmoid(self.gender_classifier(features))
        age_output = self.age_regressor(features)
        return gender_output, age_output
        
    def on_train_epoch_start(self) -> None:
        """
        Callback called before each epoch.
        """
        if self.current_epoch < self.initial_epochs:
            self.freeze_base_layers()
            self.gender_classifier.requires_grad = True
            self.age_regressor.requires_grad = True
            logging.info("Freezing all base layers...")
        else:
            self.unfreeze_base_layers()
            self.gender_classifier.requires_grad = True
            self.age_regressor.requires_grad = True
            logging.info(f"Unfreezing layers in {self.unfreeze_layers_list} for fine-tuning...")

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Training for epoch {self.current_epoch} with {num_trainable_params} trainable parameters.")

    def training_step(self, batch: Tuple[torch.Tensor, 
                      Tuple[torch.Tensor, torch.Tensor]], 
                      batch_idx: int) -> dict:
        """
        Training step for a single batch.

        Args:
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): 
            Input batch containing images, age targets, and gender targets.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the joint loss, 
            gender predictions, age predictions, and targets, along with the loss.
        """
        x, (age_target, gender_target) = batch
        gender_pred, age_pred = self(x)

        if self.gender_class_weights is not None:
            gender_loss = F.binary_cross_entropy(gender_pred.squeeze(), 
                                                 gender_target.float(), 
                                                 weight=self.gender_class_weights)
        else:
            gender_loss = F.binary_cross_entropy(gender_pred.squeeze(), 
                                                 gender_target.float())

        age_loss = F.l1_loss(age_pred.squeeze(), age_target.float())

        consistency_loss = F.mse_loss(gender_pred.squeeze(), age_pred.squeeze().sigmoid())
        joint_loss = gender_loss + age_loss + self.consistency_loss_weight * consistency_loss

        gender_preds = (gender_pred > 0.5).int()
        gender_acc = accuracy_score(gender_target.cpu().detach().numpy(), 
                                    gender_preds.cpu().detach().numpy())
        gender_f1 = f1_score(gender_target.cpu().detach().numpy(), 
                             gender_preds.cpu().detach().numpy())

        age_r2 = r2_score(age_target.cpu().detach().numpy(), 
                          age_pred.cpu().detach().numpy())
        age_mse = F.mse_loss(age_pred.squeeze(), age_target.float()).item()

        self.log('train_gender_bce_loss', gender_loss, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_age_mae_loss', age_loss, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss', joint_loss, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_gender_acc', gender_acc, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_gender_f1', gender_f1, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_age_r2', age_r2, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_age_mse', age_mse, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)

        return {'loss': joint_loss, 'gender_pred': gender_pred, 'age_pred': age_pred, 
                'gender_target': gender_target, 'age_target': age_target}

    def validation_step(self, batch: Tuple[torch.Tensor, Tuple[torch.Tensor, 
                        torch.Tensor]], batch_idx: int) -> dict:
        """
        Validation step for a single batch.

        Args:
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): 
            Input batch containing images, age targets, and gender targets.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the joint validation loss, 
            gender predictions, age predictions, and targets.
        """
        with torch.no_grad():
            x, (age_target, gender_target) = batch
            gender_pred, age_pred = self(x)

            gender_loss = F.binary_cross_entropy(gender_pred.squeeze(), gender_target.float())
            age_loss = F.l1_loss(age_pred.squeeze(), age_target.float())

            consistency_loss = F.mse_loss(gender_pred.squeeze(), age_pred.squeeze().sigmoid())
            joint_val_loss = gender_loss + age_loss + self.consistency_loss_weight * consistency_loss

            gender_preds = (gender_pred > 0.5).int()
            gender_acc = accuracy_score(gender_target.cpu(), gender_preds.cpu())
            gender_f1 = f1_score(gender_target.cpu(), gender_preds.cpu())

            age_r2 = r2_score(age_target.cpu(), age_pred.cpu())
            age_mse = F.mse_loss(age_pred.squeeze(), age_target.float()).item()

            self.log('val_gender_bce_loss', gender_loss, prog_bar=True, 
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_age_mae_loss', age_loss, prog_bar=True, 
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_loss', joint_val_loss, prog_bar=True, 
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_gender_acc', gender_acc, prog_bar=True, 
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_gender_f1', gender_f1, prog_bar=True, 
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_age_r2', age_r2, prog_bar=True, 
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_age_mse', age_mse, prog_bar=True, 
                     on_step=False, on_epoch=True, sync_dist=True)
        
        return {'val_loss': joint_val_loss, 'gender_pred': gender_pred, 
                'age_pred': age_pred, 'gender_target': gender_target, 'age_target': age_target}

   
    def on_train_epoch_end(self) -> None:
       """
       Callback called after each training epoch.
       """
       avg_train_joint_loss = self.trainer.callback_metrics['train_loss'].mean()
       avg_train_gender_loss = self.trainer.callback_metrics['train_gender_bce_loss'].mean()
       avg_train_age_loss = self.trainer.callback_metrics['train_age_mae_loss'].mean()
       avg_train_gender_acc = self.trainer.callback_metrics['train_gender_acc'].mean()
       avg_train_gender_f1 = self.trainer.callback_metrics['train_gender_f1'].mean()
       avg_train_age_r2 = self.trainer.callback_metrics['train_age_r2'].mean()
       avg_train_age_mse = self.trainer.callback_metrics['train_age_mse'].mean()

       num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
       logging.info(f"Epoch {self.current_epoch}. Trainable Parameters: {num_trainable_params}")
       logging.info(f"avg_train_joint_loss: {avg_train_joint_loss.item()}")
       logging.info(f"avg_train_gender_bce_loss: {avg_train_gender_loss.item()}")
       logging.info(f"avg_train_age_mae_loss: {avg_train_age_loss.item()}")
       logging.info(f"avg_train_gender_acc: {avg_train_gender_acc.item()}")
       logging.info(f"avg_train_gender_f1: {avg_train_gender_f1.item()}")
       logging.info(f"avg_train_age_r2: {avg_train_age_r2.item()}")
       logging.info(f"avg_train_age_mse: {avg_train_age_mse.item()}")
       self.train_losses.append(avg_train_joint_loss.item())

    def on_validation_epoch_end(self) -> None:
       """
       Callback called after each validation epoch.
       """
       if self.current_epoch_counter > 0:
           avg_val_joint_loss = self.trainer.callback_metrics['val_loss'].mean()
           avg_val_gender_loss = self.trainer.callback_metrics['val_gender_bce_loss'].mean()
           avg_val_age_loss = self.trainer.callback_metrics['val_age_mae_loss'].mean()
           avg_val_gender_acc = self.trainer.callback_metrics['val_gender_acc'].mean()
           avg_val_gender_f1 = self.trainer.callback_metrics['val_gender_f1'].mean()
           avg_val_age_r2 = self.trainer.callback_metrics['val_age_r2'].mean()
           avg_val_age_mse = self.trainer.callback_metrics['val_age_mse'].mean()

           logging.info(f"avg_val_joint_loss: {avg_val_joint_loss.item()}")
           logging.info(f"avg_val_gender_bce_loss: {avg_val_gender_loss.item()}")
           logging.info(f"avg_val_age_mae_loss: {avg_val_age_loss.item()}")
           logging.info(f"avg_val_gender_acc: {avg_val_gender_acc.item()}")
           logging.info(f"avg_val_gender_f1: {avg_val_gender_f1.item()}")
           logging.info(f"avg_val_age_r2: {avg_val_age_r2.item()}")
           logging.info(f"avg_val_age_mse: {avg_val_age_mse.item()}")
           self.val_losses.append(avg_val_joint_loss.item())
       else:
           logging.info("No validation metrics for this epoch.")
       self.current_epoch_counter += 1

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss'
            }
        }

    def freeze_base_layers(self) -> None:
        """
        Freeze the parameters of the base feature extractor.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_base_layers(self) -> None:
        """
        Unfreeze the parameters of the specified layers in the base feature extractor.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
        if self.unfreeze_layers_list is not None:
            for name, child in self.feature_extractor.named_children():
                if name in self.unfreeze_layers_list:
                    for params in child.parameters():
                        params.requires_grad = True