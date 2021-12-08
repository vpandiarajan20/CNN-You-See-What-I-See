import torch

class ModelWrapper_Clean():
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.activations = {}
        self.gradients
        for layer in layers:
            layer_hook = dict([*self.model.named_modules()])[layer]
            layer_hook.register_forward_hook(self.save_features(layer))
    
    def save_features(self, layer):
        def hook_to_register(_, __, output):
            self.activations[layer] = output
        return hook_to_register
    
    def save_gradient(self, grad):
        self.gradients = grad
        
    def get_gradients(self, classNumber, layer):
        activation = self.activations[layer]
        activation.register_hook(self.save_gradient)
        logit = self.output[:, classNumber]
        logit.backward(torch.ones_like(logit), retain_graph=True)
        gradients = self.gradients.cpu().detach().numpy()
        return gradients

    def forward(self, x):
        self.output = self.model(x)
        return self._features


