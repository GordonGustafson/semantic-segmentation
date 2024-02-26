from dataloader import *
from evaluation import confusion_matrix, f1_score, precision_and_recall

from torch import optim, nn
import torchvision.transforms as T
import lightning as L

# TODO: figure out the correct value for this.
# 31 is the largest value I've seen so far, but haven't looked very hard.
NUM_CLASSES = 32


# define the LightningModule
class PixelWiseSegmentation(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.confusion_matrices = []

    def forward(self, batch, batch_idx):
        images = batch["image"]                             # shape = (N, Channels, H, W)
        predicted_class_probabilities = self.model(images)  # shape = (N, Classes, H, W)
        return predicted_class_probabilities

    def training_step(self, batch, batch_idx):
        predicted_class_probabilities = self.forward(batch, batch_idx)  # shape = (N, Classes, H, W)
        gt_segmentation_masks = batch["segmentation_mask"]              # shape = (N, H, W)
        train_loss = nn.functional.cross_entropy(predicted_class_probabilities, gt_segmentation_masks)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        predicted_class_probabilities = self.forward(batch, batch_idx)  # shape = (N, Classes, H, W)
        gt_segmentation_masks = batch["segmentation_mask"]              # shape = (N, H, W)
        val_loss = nn.functional.cross_entropy(predicted_class_probabilities, gt_segmentation_masks)
        self.log("val_loss", val_loss)

        predicted_masks = torch.max(predicted_class_probabilities, dim=1)  # shape = (N, H, W)
        confusion_mat = confusion_matrix(gt_segmentation_masks.numpy(), predicted_masks.numpy())
        self.confusion_matrices.append(confusion_mat)

    def on_validation_epoch_end(self) -> None:
        val_set_confusion_matrix = sum(self.confusion_matrices)

        print(val_set_confusion_matrix)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999))


# init the model
conv = nn.Conv2d(in_channels=3, out_channels=NUM_CLASSES, kernel_size=1, padding=0, stride=1)
model = PixelWiseSegmentation(conv)

# setup data
transform = image_transform_to_dict_transform(T.ToTensor())
# train_dataloader = get_mini_dataloader(batch_size=2, transform=transform)
# train_dataloader = get_dataloaders(batch_size=2, train_transform=transform, val_transform=transform)["train"]
train_dataloader = get_mini_dataloader(batch_size=2, transform=transform)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=train_dataloader)
