import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class DeepFM(nn.Module):
    def __init__(self, num_features, num_factors, num_deep_layers=20, deep_layer_size=128, lr=0.01, weight_decay=0.01):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.weight_decay = weight_decay
        
        # FM part
        self.w = nn.Parameter(torch.randn(num_features))
        self.v = nn.Parameter(torch.randn(num_features, num_factors))
        
        # Deep part
        input_size = num_features  # Adjust this line to match the shape of your input data
        self.deep_layers = nn.ModuleList()
        for i in range(num_deep_layers):
            self.deep_layers.append(nn.Linear(input_size, deep_layer_size))
            self.deep_layers.append(nn.ReLU())
            input_size = deep_layer_size
        
        self.deep_output_layer = nn.Linear(input_size, 1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        #self.loss_func = nn.MSELoss()
        
    def forward(self, x):
        # FM part
        linear_terms = torch.matmul(x, self.w)
        interactions = 0.5 * torch.sum(
            torch.matmul(x, self.v) ** 2 - torch.matmul(x ** 2, self.v ** 2),
            dim=1,
            keepdim=True
        )
        
        # Deep part
        deep_x = x  # No need to flatten
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
        deep_out = self.deep_output_layer(deep_x)
        
        return linear_terms + interactions.squeeze() + deep_out.squeeze()


    def loss(self, y_pred, y_true, c_values):
        mse = (y_pred - y_true.float()) ** 2
        weighted_mse = c_values * mse
        l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2  # L2 regularization
        return torch.mean(weighted_mse) + self.weight_decay * l2_reg
        
    # def loss(self, y_pred, y_true):
    #     return self.loss_func(y_pred, y_true.float())

    def train_step(self, x, y,c_values_tensor):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y,c_values_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class EnsembleFM:
    def __init__(self, num_features, num_factors, lr=0.01, weight_decay=0.01):
        self.fm = FactorizationMachine(num_features, num_factors, lr, weight_decay)
        self.deepfm = FactorizationMachine(num_features, num_factors, lr, weight_decay)
        
    def fit(self, X, y, num_epochs=10):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        for epoch in range(num_epochs):
            fm_loss = self.fm.train_step(X_tensor, y_tensor)
            deepfm_loss = self.deepfm.train_step(X_tensor, y_tensor)
            print(f'Epoch {epoch+1}/{num_epochs}, FM Loss: {fm_loss:.4f}, DeepFM Loss: {deepfm_loss:.4f}')
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            fm_pred = self.fm(X_tensor)
            deepfm_pred = self.deepfm(X_tensor)
        return fm_pred, deepfm_pred
    
    def optimize_ensemble_weights(self):
        def objective(weights):
            ensemble_pred = weights[0] * fm_pred + weights[1] * deepfm_pred
            return nn.MSELoss()(ensemble_pred, y_tensor.float())
        
        fm_pred, deepfm_pred = self.predict(X)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        res = gp_minimize(objective, [(0.0, 1.0), (0.0, 1.0)], n_calls=20)
        optimal_weights = res.x
        return optimal_weights