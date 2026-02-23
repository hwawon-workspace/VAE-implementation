import torch
import os
import torchvision.utils as vutils

class VAEtest:
    def __init__(self, model, model_save_path, device, channels, image_save_path, filename):
        self.model = model
        self.model.load_state_dict(torch.load(model_save_path))
        self.model = self.model.to(device)
        self.device = device
        self.channels = channels
        self.image_save_path = image_save_path
        self.filename = filename

    def BCE_KLD(self, recon_x, x, mu, log_var):  # 흑백 이미지의 경우
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def MSE_KLD(self, recon_x, x, mu, log_var):  # 컬러 이미지의 경우
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def get_loss_function(self):
        return self.BCE_KLD if self.channels == 1 else self.MSE_KLD

    def test(self, test_loader, save_image=True):
        self.model.eval()
        test_loss = 0
        loss_function = self.get_loss_function()
        image_saved = False

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                test_loss += loss.item()

                # 첫 번째 배치만 이미지 저장
                if save_image and not image_saved:
                    self.save_images(data, recon_batch)
                    image_saved = True

            avg_test_loss = test_loss / len(test_loader.dataset)
            print(f'Avg Test loss: {avg_test_loss:.4f}')
            return avg_test_loss

    def save_images(self, input_images, recon_images):
        filename = f'{self.filename}.png'
        save_path = os.path.join(self.image_save_path, filename)
        
        # 중복된 파일명 유무 확인
        save_path = self.get_unique_filename(save_path)
        images_to_save = torch.cat((input_images[:4], recon_images[:4]), dim=2).to(self.device)
        vutils.save_image(images_to_save, save_path, normalize=True)
        print(f"이미지가 {save_path}에 저장되었습니다.")
        
    def get_unique_filename(self, path):
        base, extension = os.path.splitext(path)
        counter = 1
        while os.path.exists(path):
            path = f'{base}_{counter}{extension}'
            counter += 1
        return path
# %%
