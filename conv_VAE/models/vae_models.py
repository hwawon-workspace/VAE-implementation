# 종류별 VAE모델을 클래스로 정의
#%%
import torch
import torch.nn as nn
import math
#%%
# VAE(Linear layer로 구성)
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, hidden_dims):
        # input_shape은 datasets.py 에서 데이터셋 모듈에서 정의됨. (C, H, W)
        # latent_dim, hidden_dims는 config 파일에서 정의됨.
        super(VAE, self).__init__()
        self.channels, self.height, self.width = input_shape
        self.input_dim = self.channels*self.height*self.width
        
        # Encoder
        en = []
        in_dim = self.input_dim
        for h in hidden_dims:
            en.append(nn.Linear(in_dim, h))
            en.append(nn.ReLU())
            in_dim = h
        en.append(nn.Linear(h, latent_dim * 2)) # mu, logvar 공간을 마련하기 위해 *2 해줌
        # 신경망을 self.encoder 속성으로 정의
        self.encoder = nn.Sequential(*en)
        
        # Decoder
        de = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            de.append(nn.Linear(in_dim, h))
            de.append(nn.ReLU())
            in_dim = h
        de.append(nn.Linear(h, self.input_dim))
        de.append(nn.Sigmoid())
        # 신경망을 self.decoder 속성으로 정의
        self.decoder = nn.Sequential(*de)
    
    # 인코딩
    def encode(self, x):
        x = x.view(-1, self.input_dim) # x shape을 (batch_size, input_dim)으로 변환
        mu_logvar = self.encoder(x)
        return mu_logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    # 디코딩
    def decode(self, z):
        decoded = self.decoder(z)
        decoded = decoded.view(-1, self.channels, self.height, self.width) # shape을 다시 x shape (batch_size, C, H, W)로 변환
        return decoded
        
    def forward(self, x):
        mu_logvar = self.encode(x)
        mu, logvar = mu_logvar.chunk(2, dim = 1) # mu, logvar를 각각 나눠줌
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
        
#%%        
# convVAE(Conv2d layer로 구성)       
class convVAE(nn.Module):
    def __init__(self, input_shape, latent_dim, hidden_dims):
        super(convVAE, self).__init__()
        self.channels, self.height, self.width = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        en = []
        outputpaddings = []
        in_dim = self.channels
        self.red_h = self.height
        self.red_w = self.width
        for h in hidden_dims:
            en.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = in_dim,
                              out_channels = h,
                              kernel_size = 3,
                              stride = 2,
                              padding = 1),
                    nn.BatchNorm2d(h),
                    nn.LeakyReLU()
                )
            )
            in_dim = h
            outputpaddings.append(0) if (self.red_h-1)%2 == 0 else outputpaddings.append(1)
            self.red_h = math.floor((self.red_h-1)/2) + 1
            self.red_w = math.floor((self.red_w-1)/2) + 1

        self.encoder = nn.Sequential(*en)
        # red는 Conv2d layer를 거쳐 축소된 크기
        # kernel = 3, stride = 2, padding = 1일 때 output의 크기는 2배씩 줄어듦
        # hidden dim의 개수만큼 Conv2d layer를 거치므로 2**(hidden dim 개수)만큼 줄어듦
        self.hidden_ch = h
        self.flat_dim = self.hidden_ch * self.red_h * self.red_w # 마지막 hidden dim 크기(채널 수) * 줄어든 feature map 크기(H*W)
        self.en_fc = nn.Linear(self.flat_dim, latent_dim * 2)
        
        # Decoder
        self.de_fc = nn.Linear(latent_dim, self.flat_dim)
        de = []
        in_dim = self.hidden_ch
        for h, outpad in zip(reversed([self.channels]+hidden_dims[:-1]), reversed(outputpaddings)):
            de.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = in_dim,
                        out_channels = h,
                        kernel_size = 3,
                        stride = 2,
                        padding = 1,
                        output_padding= outpad    
                    ),
                    nn.BatchNorm2d(h),
                    nn.LeakyReLU()
                )
            )
            in_dim = h
        de.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*de)
     
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(-1, self.flat_dim) # en_fc에 넣기 위해 flatten
        mu_logvar = self.en_fc(encoded)
        return mu_logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        decoded = self.de_fc(z)
        decoded = decoded.view(-1, self.hidden_ch, self.red_h, self.red_w) # decode에 넣기 위해 reshape
        decoded = self.decoder(decoded)
        return decoded
    
    def forward(self, x):
        mu_logvar = self.encode(x)
        mu, logvar = mu_logvar.chunk(2, dim = 1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
# #%%
# model = convVAE((1, 28, 28), 20, [32, 64, 128])
# print(model)

# # vqVAE
# # %%
# print(model.hidden_ch)
# print(model.red_h, model.red_w)
# print(model.flat_dim)
# print(model.en_fc)
# # %%
# print(model.outputpaddings)
# %%
