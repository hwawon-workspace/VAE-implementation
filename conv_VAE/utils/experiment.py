import torch
import torch.nn.functional as F
from tqdm import tqdm

class VAExperiment():
    def __init__(self, model, device, channels, model_save_path):
        self.model = model
        self.device = device
        self.channels = channels
        self.model_save_path = model_save_path
        self.best_val_loss = float('inf')

    # loss function 정의
    def BCE_KLD(self, recon_x, x, mu, log_var): # 이미지가 [0, 1] 범위로 정규화된 경우
        BCE = F.binary_cross_entropy(recon_x, x, reduction = 'sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def MSE_KLD(self, recon_x, x, mu, log_var): # 각 채널이 연속적인 값을 가질 때
        MSE = F.mse_loss(recon_x, x, reduction = 'sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD
    
    def get_loss_function(self):
        return self.BCE_KLD if self.channels == 1 else self.MSE_KLD
    
    # train 메서드
    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        train_loss = 0
        loss_function = self.get_loss_function()
        
        for data, _ in tqdm(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"====> Epoch: {epoch}\nAvg Train loss: {avg_train_loss:.4f}")
        return avg_train_loss

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        loss_function = self.get_loss_function()
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Avg Valid loss: {avg_val_loss:.4f}")
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f'new best model saved')
        return avg_val_loss