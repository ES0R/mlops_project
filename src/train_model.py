import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
import hydra
from omegaconf import DictConfig
from models.model import CustomCNN, MyImprovedCNNModel  # Adjust path according to your project structure
from lightning.pytorch import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
import datetime
import omegaconf


# Suppress specific warning messages
warnings.filterwarnings("ignore", category=UserWarning, module="all")
warnings.filterwarnings("ignore", message=".*Consider setting `persistent_workers=True`.*")
warnings.filterwarnings("ignore", message=".*You are using a CUDA device.*")

class ImageClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super(ImageClassifier, self).__init__()
        self.cfg = cfg
        model_cfg = getattr(cfg.models, cfg.default_model)  
        self.model = MyImprovedCNNModel()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
            images, targets = batch
            outputs = self(images)
            loss = self.criterion(outputs, targets)
            acc = self.calculate_accuracy(outputs, targets)  # Implement this method
            self.log('train_loss', loss)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        acc = self.calculate_accuracy(outputs, targets)  # Implement this method
        self.log('val_loss', loss)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def calculate_accuracy(self, outputs, targets):
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.cfg.hyperparameters.lr, weight_decay=self.cfg.hyperparameters.wd)
        return optimizer
    
    # def configure_logging(self):
    # wandb_logger = WandbLogger(
    #     log_model=True,  # Log the best model
    #     save_dir="wandb_logs",  # Save logs in a specific directory
    # )
    # return [wandb_logger]

def get_run_name(cfg):
    model_name = cfg.models.base.name
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_{current_time}"

def load_data(cfg):
    train_images_tensor = torch.load(cfg.data.image_data)
    train_target_tensor = torch.load(cfg.data.target_data)
    assert len(train_images_tensor) == len(train_target_tensor), "Mismatch in data length"

    full_dataset = TensorDataset(train_images_tensor, train_target_tensor)
    train_size = int(cfg.data.train_val_split[0] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.model.hyperparameters.batch_size, shuffle=True, num_workers=cfg.data.num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.model.hyperparameters.batch_size, shuffle=False, num_workers=cfg.data.num_workers, persistent_workers=True)
    return train_loader, val_loader




@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    seed_everything(42, workers=True)

    torch.set_float32_matmul_precision('high')  # or 'medium', depending

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Run name
    run_name = get_run_name(cfg.model)


    # Set up Wandb Logger
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(name="Training_Run", project="MLOps-Project", config=wandb_config)

    # Load data
    train_loader, val_loader = load_data(cfg)

    # Initialize model
    model = ImageClassifier(cfg.model)

    # Check points
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(hydra.utils.get_original_cwd(), f"checkpoints/{run_name}/"),
    monitor='val_acc',
    mode='max',
    save_top_k=1,
    filename='{epoch}-{val_acc:.2f}'
    )

    #Set up PyTorch Lightning trainer
    trainer = pl.Trainer(
        logger=wandb_logger, 
        max_epochs=cfg.model.hyperparameters.epochs,
        deterministic=True,
        accelerator="auto",  
        callbacks=[checkpoint_callback],
        enable_checkpointing=True, #TODO ---------------------------------  
        enable_progress_bar=True,  
        log_every_n_steps=10  
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Optionally, save your trained model
    model_path = os.path.join(hydra.utils.get_original_cwd(), f'models/trained_model_{cfg.model.models.base.name}.pth')
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train()
