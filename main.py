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
    model = load_model()
    tcav = TCAV(model, class_number, ['feature_layers'], "zebras_from_kaggle")
    # tcav.generate_cavs_sklearn_class("Striped", "RandomImages")
    # score = tcav.return_tcav_score()
    # print("TCAV Score Scikit-Learn SGDClassifier:", score)

    # tcav.generate_cavs_sklearn_logreg("Striped", "RandomImages")
    # score = tcav.return_tcav_score()
    # print("TCAV Score Scikit-Learn Logistic Regression:", score)

    tcav.generate_cavs_pytorch_class("Striped", "RandomImages")
    score = tcav.return_tcav_score()
    print("TCAV Score Pytorch Classifier:", score)

    


if __name__ == '__main__':
    main()
