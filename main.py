from TCAV_Clean import TCAV
import torchvision.models as models
from model import Resnet18
class_number = 463
conceptFileName = "ZigZag"

def load_model():
    '''
    Loads pretrained model on imagenet from pytorch repository
        Parameters: 
            None
        Returns: 
            model (pytorch model): pytorch model trained on imagnet 
    '''
    return Resnet18()

def main():
    model = load_model()
    tcav = TCAV(model, class_number, ['feature_layers'], 'zebras_from_kaggle')

    # tcav.generate_cavs_sklearn_class(conceptFileName, "RandomImages")
    # score = tcav.compute_tcav_score(tcav.cavs[0], tcav.layers[0])
    # print("TCAV Score Scikit-Learn SGDClassifier:", score)

    # tcav.generate_cavs_sklearn_logreg(conceptFileName, "RandomImages")
    # score = tcav.compute_tcav_score(tcav.cavs[1], tcav.layers[0])
    # print("TCAV Score Scikit-Learn LogReg:", score)

    # tcav.generate_cavs_pytorch_class(conceptFileName, "RandomImages")
    # score = tcav.compute_tcav_score(tcav.cavs[2], tcav.layers[0])
    # print("TCAV Score Pytorch:", score)

    print("TCAV Score SGD:", tcav.run_tcav(conceptFileName, "RandomImages", 'feature_layers', "SGD"))

    tcav2 = TCAV(model, class_number, ['feature_layers'], 'zebras_from_kaggle')

    print("TCAV Score Linear:", tcav2.run_tcav(conceptFileName, "RandomImages", 'feature_layers', "LINEAR"))

if __name__ == '__main__':
    main()
