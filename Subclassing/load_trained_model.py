
def load_ensemble(model) : 
    """
    Note that ensemble model differs by submodels, 
    which means every ensemble model could be different by how submodels are implanted and how many submodels are in ensemble model.

    Eventually, this function could be useless, if different submodels are implanted.

    Implanted submodels : "LeNet5", "VGG16"
    """
    
    model.model_list[0].load_weights("Subclassing/model_checkpoints/Ensemble/Submodel_checkpoint/LeNet5_01_0.2740.hdf5")
    model.model_list[1].load_weights("Subclassing/model_checkpoints/VGG16/53_0.5009.hdf5")
    model.__model__.load_weights("Subclassing/model_checkpoints/Ensemble/Supmodel_checkpoint/ensemble_06_0.2691.hdf5")
    return model

def load_ensemble_sup(model) : 
    """
    Note that ensemble model differs by submodels, 
    which means every ensemble model could be different by how submodels are implanted and how many submodels are in ensemble model.

    Eventually, this function could be useless, if different submodels are implanted.

    Implanted submodels : "LeNet5", "VGG16"
    """

    model.__model__.load_weights("Subclassing/model_checkpoints/Ensemble/Supmodel_checkpoint/ensemble_06_0.2691.hdf5")
    return model

def load_trained_googlenet(model) : 
    model.load_weights("Subclassing/model_checkpoints/GoogLeNet/22_0.9526.hdf5")
    return model

def load_trained_lenet5(model) : 
    model.load_weights("Subclassing/model_checkpoints/LeNet5/21_0.4171.hdf5")
    return model

def load_trained_vgg16(model) : 
    model.load_weights("Subclassing/model_checkpoints/VGG16/53_0.5009.hdf5")
    return model

def load_model(model, filepath) : 
    model.load_weights(filepath)
    return model

