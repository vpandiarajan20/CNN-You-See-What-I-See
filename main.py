from TCAV_Clean import TCAV
import torchvision.models as models
from model import Resnet18
class_number = 463

def load_model():
    '''
    Loads pretrained model on imagenet from pytorch repository
        Parameters: 
            None
        Returns: 
            model (pytorch model): pytorch model trained on imagnet 
    '''
    return Resnet18()
    #return models.resnet101(pretrained=True) 

def main():
    model1 = load_model()
    tcav1 = TCAV(model1, class_number, ['feature_layers'], "zebras_from_kaggle")
    tcav1.generate_cavs_sklearn_logreg("Striped", "RandomImages")
    score = tcav1.return_tcav_score()
    print("TCAV Score Scikit-Learn Logistic Regression:", score)

    model2 = load_model()
    tcav2 = TCAV(model2, class_number, ['feature_layers'], "zebras_from_kaggle")
    tcav2.generate_cavs_sklearn_class("Striped", "RandomImages")
    score = tcav2.return_tcav_score()
    print("TCAV Score Scikit-Learn SGDClassifier:", score)

    model3 = load_model()
    tcav3 = TCAV(model3, class_number, ['feature_layers'], "zebras_from_kaggle")
    tcav3.generate_cavs_pytorch_class("Striped", "RandomImages")
    score = tcav3.return_tcav_score()
    print("TCAV Score Scikit-Learn Pytorch Classifier:", score)

    # tcav.generate_cavs_pytorch_class("Dotted", "RandomImages")
    # score = tcav.return_tcav_score()
    # print("TCAV Score Pytorch Classifier:", score)

    


if __name__ == '__main__':
    main()
