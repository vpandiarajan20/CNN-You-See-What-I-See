import torch
from torch import nn

class ModelWrapper_Clean(nn.Module):
    # def __init__(self, model, layers):
    #     super().__init__()
    #     self.model = model
    #     self.layers = layers
    #     self.activations = {layer: torch.empty(0) for layer in layers}

    #     for layer in layers:
    #         layer_hook = dict([*self.model.named_modules()])[layer]
    #         layer_hook.register_forward_hook(self.save_features(layer))
    
    # def save_features(self, layer):
    #     def hook_to_register(_, __, output):
    #         self.activations[layer] = output
    #     return hook_to_register
    
    # def save_gradient(self, grad):
    #     self.gradients = grad
        
    # def get_gradients(self, classNumber, layer):
    #     activation = self.activations[layer]
    #     activation.register_hook(self.save_gradient)
    #     logit = self.output[:, classNumber]
    #     logit.backward(torch.ones_like(logit), retain_graph=True)
    #     gradients = self.gradients.cpu().detach().numpy()
    #     return gradients

    # def forward(self, x):
    #     self.output = self.model(x)
    #     return self.activations



    # def __init__(self, model, layers):
    #     super().__init__()
    #     self.model = model
    #     self.layers = layers
    #     self._features = {layer: torch.empty(0) for layer in layers}

    #     for layer_id in layers:
    #         layer = dict([*self.model.named_modules()])[layer_id]
    #         layer.register_forward_hook(self.save_outputs_hook(layer_id))

    # def save_outputs_hook(self, layer_id):
    #     def fn(_, __, output):
    #         self._features[layer_id] = output
    #     return fn
    
    # def save_grads(self, grad):
    #     self.gradients = grad

    # def get_gradients(self, className, layerName):
    #     activations = self._features[layerName]

    #     activations.register_hook(self.save_grads)
    #     logits = self.output[:, className]

    #     logits.backward(torch.ones_like(logits), retain_graph = True)
    #     gradients = self.gradients.cpu().detach().numpy()
    #     return gradients

    # def forward(self, x):
    #     self.output = self.model(x)
    #     return self._features



    def __init__(self, model, layers):
        super().__init__()
        # self.model = deepcopy(model)
        self.model = model
        self.intermediate_activations = {}

        def save_activation(name):
            '''create specific hook by module name'''

            def hook(module, input, output):
                self.intermediate_activations[name] = output

            return hook

        for name, module in self.model._modules.items():
            if name in layers:
                # register the hook
                module.register_forward_hook(save_activation(name))

    def save_gradient(self, grad):
        self.gradients = grad

    def get_gradients(self, c, layer_name):
        activation = self.intermediate_activations[layer_name]
        activation.register_hook(self.save_gradient)
        logit = self.output[:, c]
        logit.backward(torch.ones_like(logit), retain_graph=True)
        # gradients = grad(logit, activation, retain_graph=True)[0]
        # gradients = gradients.cpu().detach().numpy()
        gradients = self.gradients.cpu().detach().numpy()
        return gradients

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x):
        self.output = self.model(x)
        return self.intermediate_activations