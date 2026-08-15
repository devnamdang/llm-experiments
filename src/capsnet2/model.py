import torch
import torch.nn as nn

"""
This a MNIST capsnet model loosely based on [1]. Might not be 100% the same. 

[1]: Sabour and Frosst (2017), Dynamic Routing Between Capsules,
"""

config = {
    "im_height": 32,
    "im_width": 32,
    "im_channels": 3,
    "conv_out_channels": 256,
    "primary_caps_dim": 16,
    "n_primary_caps": 32,
    "digit_caps_dim": 32,
    "route_iter": 4,
    "n_classes": 10
}


def softmax_squash(v, eps=1e-8):
    """
    Args:
        v torch.tensor: Tensor of shape (batch_size, n_vectors, vector_dim)
    Returns:
        torch.tensor of shape (batch_size, n_vectors, vector_dim)
    """
    squared_norm = v.square().sum(dim=-1, keepdim=True)
    safenorm = torch.sqrt(squared_norm + eps) # Use safenorm to prevent division by zero
    softmax = nn.Softmax(dim=1)
    v_squashed = softmax(safenorm) * v / safenorm
    return v_squashed


class Weighting(nn.Module):
    def __init__(self, n_caps, in_cap_dim, out_cap_dim, n_classes):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, n_caps, in_cap_dim, out_cap_dim*n_classes))
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x torch.Tensor: Tensor of size (batch_size, n_caps, caps_dim)
        
        Returns:
            torch.Tensor: Tensor of size (batch_size, n_caps, output_cap_dim*n_classes)
        """
        assert len(x.shape)==3, "Dimension of input tensor has to be 3"
        x = x.unsqueeze(dim=-2) # (batch_size, n_caps, caps_dim) -> (batch_size, n_caps, 1, caps_dim)
        y = x @ self.w # (batch_size, n_caps, 1, caps_dim) @ (1, n_caps, in_cap_dim, out_cap_dim*n_classes) -> (batch_size, n_caps, 1, out_cap_dim*n_classes)
        y = y.squeeze(dim=-2) # (batch_size, n_caps, 1, out_cap_dim*n_classes) -> (batch_size, n_caps, out_cap_dim*n_classes)
        return y

def squash(x, eps=1e-8):
    """
    Returns squashed vectors:
    |x|^2 / (1+|x|^2) * 1/safenorm(x),
    where safenorm(x) = sqrt(sum(x^2) + eps)
    """
    squared_norm = x.square().sum(dim=-1, keepdim=True)
    r_safenorm = torch.rsqrt(squared_norm + eps)
    x_squashed = (squared_norm / (1+squared_norm)) * x * r_safenorm
    return x_squashed

def route(u, n_iter):
    """
    Arguments:
        u (tensor): Tensor with shape (n_digits, n_all_caps, digit_dim)
        n_iter (int): Number of iterations
    
    Returns:
        Next capsule layer
    """
    batch_size, n_digits, n_all_caps, _ = u.shape
    b = torch.zeros(batch_size, n_digits, n_all_caps) # Shape (batch_size, n_digits, n_all_caps)
    for i in range(n_iter):
        c = b.softmax(dim=-1) # Shape (batch_size, n_digits, n_all_caps)
        s = c.unsqueeze(dim=-2) @ u # Shape (batch_size, n_digits, 1, digit_dim)
        v = squash(s) # Shape (batch_size, n_digits, 1, digit_dim)
        if i < n_iter-1:
            delta_b = u @ v.transpose(dim0=-1, dim1=-2) # Shape: (batch_size, n_digits, n_all_caps, digit_dim) @ (batch_size, n_digits, digit_dim, 1) -> (batch_size, n_digits, n_all_caps, 1) 
            b += delta_b.squeeze(dim=-1) # Shape (batch_size, n_digits, n_all_caps)
    return v

class Decoder(nn.Module):
    def __init__(self, in_features, final_w = 28, out_channels=3):
        super().__init__()
        self.w = final_w - 22 # w=10 for 32x32 images
        self.fc_end = nn.Linear(in_features, 8 * (self.w**2))
        self.bn = nn.BatchNorm1d(8 * (self.w**2), momentum=0.8)
        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(8, 128, 3),
                nn.ConvTranspose2d(128, 64, 5),
                nn.ConvTranspose2d(64, 32, 5),
                nn.ConvTranspose2d(32, 16, 5),
                nn.ConvTranspose2d(16, 16, 5),
                nn.ConvTranspose2d(16, 16, 3),
                nn.ConvTranspose2d(16, out_channels, 3)
                )
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        # Get highest norms vectors
        x_norm = x.norm(dim=-1, p=2, keepdim=False).squeeze()
        arg_max = x_norm.argmax(dim=-1, keepdim=True) # (batch_size)
        x = torch.cat([v[arg_max[idx]] for idx, v in enumerate(x)]) # (batch_size, digit_dim)
        x = x.squeeze(dim=1)
        
        # Pass digitcap classes into decoder
        x = self.fc_end(x)
        x = self.bn(x)
        x = x.reshape(x.shape[0], 8, self.w, self.w)
        x = self.sigmoid(self.deconv(x))
        return x

class CapsNet(nn.Module):
    def __init__(self, cfg=config):
        super().__init__()
        # Set attributes
        self.primary_caps_dim = cfg["primary_caps_dim"] 
        self.digit_caps_dim = cfg["digit_caps_dim"]
        self.n_classes = cfg['n_classes']
        caps_out_channels = self.primary_caps_dim * cfg["n_primary_caps"]
        # Convolutional Layers
        self.conv_1 = nn.Conv2d(in_channels=cfg['im_channels'],
                              out_channels=cfg['conv_out_channels'],
                              kernel_size=9,
                              bias=True
                            )
        self.relu = torch.nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=self.conv_1.out_channels,
                            out_channels=caps_out_channels,
                            kernel_size=9, stride=2)

        self.cap_w = 6 if cfg['im_width'] == 28 else 8
        n_total_primary_caps = cfg["n_primary_caps"] * self.cap_w * self.cap_w
        self.weighting = Weighting(n_caps=n_total_primary_caps, in_cap_dim=self.primary_caps_dim, out_cap_dim=self.digit_caps_dim, n_classes=self.n_classes)
        
        # Decoder
        self.im = {}
        self.im['height'] = cfg['im_height']
        self.im['width'] = cfg['im_width']
        self.im['channels'] = cfg['im_channels']
        #out_features = cfg['im_height'] * cfg['im_width'] * cfg['im_channels']
        self.decoder = Decoder(in_features=self.digit_caps_dim, final_w=self.im['width'], out_channels=cfg['im_channels'])

    def forward(self, x):
        # Create primary caps with two conv layers
        batch_size = x.shape[0]
        x = self.relu(self.conv_1(x)) # (batch_size, n_channels, H, W) -> (batch_size, out_channels, 20, 20)
        x = self.conv_2(x) # (batch_size, out_channels, 20, 20) -> (batch_size, primary_caps_dim*n_primary_caps, 6, 6)
        x = x.reshape(batch_size, -1, self.primary_caps_dim, self.cap_w, self.cap_w).movedim(2,-1) # (batch_size, primary_caps_dim*n_primary_caps, 6, 6) -> (batch_size, n_primary_caps, 6, 6, caps_dim)

        # Weight primary caps and route to digit caps
        x = x.flatten(start_dim=1,end_dim=-2) #(batch_size, n_primary_caps, 6, 6, caps_dim) -> (batch_size, n_total_primary_caps, caps_dim)
        u = self.weighting(x) # (batch_size, n_total_primary_caps, caps_dim) -> (batch_size, n_total_primary_caps, n_digits*digit_dim)
        u = u.reshape(batch_size, -1, self.n_classes, self.digit_caps_dim) # (batch_size, n_total_primary_caps, n_digits*digit_dim) -> (batch_size, n_total_primary_caps, n_digits, digit_dim)
        u = u.transpose(dim0=1, dim1=2) # (batch_size, n_total_primary_caps, n_digits, digit_dim) -> (batch_size, n_digits, n_total_primary_caps, digit_dim)
        
        # Digit caps
        #v = u.sum(dim=2) # (batch_size, n_digits, n_total_primary_caps, digit_dim) -> (batch_size, n_digits, digit_dim)
        #v = v.unsqueeze(dim=-2)
        #v = softmax_squash(v).unsqueeze(dim=-2) # (batch_size, n_digits, 1, digit_dim)
        v = route(u, 1) # (batch_size, n_digits, 1, digit_dim)

        # Output
        v_norm = v.norm(dim=-1, p=2, keepdim=False).squeeze(dim=-1) # (batch_size, n_digits, 1, digit_dim) -> (batch_size, n_digits)
        y = self.decoder(v) # (batch_size, n_digits, 1, digit_dim) -> (batch_size, 28*28)
        return v_norm, y


