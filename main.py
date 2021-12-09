from TCAV_Clean import TCAV
import torchvision.models as models
from model import Resnet18
class_number = 463
conceptFileName = "Striped"

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
    tcav1 = TCAV(load_model(), class_number, ['feature_layers'], 'zebras_from_kaggle')
    tcav1.generate_cavs_sklearn_class(conceptFileName, "RandomImages")
    score = tcav1.compute_tcav_score(tcav1.cavs[0], tcav1.layers[0])
    print("TCAV Score Scikit-Learn SGDClassifier:", score)

    tcav2 = TCAV(load_model(), class_number, ['feature_layers'], 'zebras_from_kaggle')
    tcav2.generate_cavs_sklearn_logreg(conceptFileName, "RandomImages")
    score = tcav2.compute_tcav_score(tcav2.cavs[0], tcav2.layers[0])
    print("TCAV Score Scikit-Learn LogReg:", score)

    tcav3 = TCAV(load_model(), class_number, ['feature_layers'], 'zebras_from_kaggle')
    tcav3.generate_cavs_pytorch_class(conceptFileName, "RandomImages")
    score = tcav3.compute_tcav_score(tcav3.cavs[0], tcav3.layers[0])
    print("TCAV Score Pytorch:", score)

    # print("TCAV Score SGD:", tcav.run_tcav(conceptFileName, "RandomImages", 'feature_layers', "SGD"))

    # tcav2 = TCAV(load_model(), class_number, ['feature_layers'], 'zebras_from_kaggle')

    # print("TCAV Score Linear:", tcav2.run_tcav(conceptFileName, "RandomImages", 'feature_layers', "LINEAR"))

if __name__ == '__main__':
    main()
