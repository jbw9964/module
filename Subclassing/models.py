
# ====================================================================================================== #
# Ensemble
# May be done

import numpy as np

from keras.layers import Input, Dense, Average
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Model

class Ensemble(Model) : 
    """
    You can't use F1ScoreCallback or callbacks that use raw datasets.
    You can't use validation_data=(x_val, y_val), since it uses raw datasets.

    Reason why you cant these : method overriding
    - In class method, there's predict method. This Ensemble model make prediction by output of submodels.
    The important is that the output of these submodels are shape=(num_data, num_classes), so inputs of supermodel is shape=(num_submodels, num_data, num_classes).
    - This is where error occurs. The input shape of supermodel has to be shape(num_submodles, num_data, num_classes). But due to method overriding, the input_shape of supermodel become shape=(num_data, raw_dataset).
    Eventually, this wrong shape make supermodel to conflict the number of inputs (num_model) as just 1
    - Sample Error code : 
        - "ValueError: Layer "Supmodel_input" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(None, 28, 28, 1) dtype=float32>]"
    """
    def __init__(self, models, num_classes, name='Ensemble', optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], *args, **kwargs):
        """
        This ensemble model consist of two model : Submodel, Supermodel
        - Submodel : 
            - The models to ensemble. 
            - Submodels must be trained before. If not, this ensemble models' performance will be very low.
            - Use "fit_submodel" to train every submodels.
        - Supermodel : 
            - The model which becomes output of ensemble model
            - Supermodel is trained by prediction of submodels. In a nutshell, supermodel trains the weights of each submodels' output, 
            and averages the weighted output. Eventually, supermodel will act like "weighted averaging".
            - Since supermodel is trained by submodels' predition (outputs), supermodel should be trained after submodels are trained. 
            If not, models' performance will be very low.
            - Use "fit_supermodel" to train supermodel individually.

        Args
        - Essential args : 
            - models        : any iterable object that contains submodels
            - num_classes   : number of labels (or target) to classification
        - Non essential args : 
            - name          : name of ensemble (super) model
            - optimizer     : optimizer to fit supermodel
            - loss          : loss to calculate with supermodel
            - metrics       : metrics to calculate with supermodel
        """
        
        super(Ensemble, self).__init__(name=name, *args, **kwargs)
        self.model_list = models
        self.num_classes = num_classes

        self.input_list = []
        for _ in range(len(self.model_list)) : 
            self.input_list.append(Input(shape=num_classes, name=self.model_list[_].name))
        
        self.dense_list = []
        for index, layer in enumerate(self.input_list) : 
            self.dense_list.append(Dense(num_classes * 3, name=self.model_list[index].name + "_Dense")(layer))

        self.average = Average(name=self.name + "_Average")(self.dense_list)
        self.output_layer = Dense(num_classes, name= self.name + "_Output", activation='softmax')(self.average)

        self.__model__ = Model(inputs=self.input_list, outputs=[self.output_layer], name=self.name)
        self.__model__.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Done
    def fit(self, x=None, y=None, batch_size=None, epochs_sub=3, epochs_sup=10, verbose=1, callbacks=[EarlyStopping(verbose=1, restore_best_weights=True, patience=5)], validation_split=0.3, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        """
        Fits submodels and supermodel with input datasets.
        Denote that this method can cause time delay of kernel or overusage of RAM, depending the submodels you implant.
        - Input datasets must be shaped as same as datasets that trained submodels.

        For example : 
        - if sub_train_datasets     : x.shape = (6800, 28, 28, 1), y.shape = (6800)
        - input_datasetes should    : x.shape = (6800, 28, 28, 1), y.shape = (6800)

        Args : 
        - Essential args : 
            - x                 : train data
            - y                 : target data
        - Util args : 
            - epochs_sub        : maximum epoch that trains submodels
            - epochs_sup        : maximum epoch that trains supermodels
            - callbacks         : Predefined or custom callbacks to use when fitting the models
            - validation_split  : portion of validation datasets (0 ~ 1)

        Return : 
            - history_of_submodels ('list'), history_of_supermodel ('history' object)
        
        Note that Ensemble model can't use validation_data=(x_val, y_val) or F1ScoreCallback due to method overriding.
        """
        
        print("================================================================================")
        print("--------------------------------------------------------------------------------")
        submodel_history = self.fit_submodel(x=x, y=y, batch_size=batch_size, epochs=epochs_sub, verbose=verbose, callbacks=callbacks, validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        print()
        print("Submodels are successfully trained")
        print()
        print("--------------------------------------------------------------------------------")
        supmodel_history = self.fit_supermodel(x=x, y=y, batch_size=batch_size, epochs=epochs_sup, verbose=verbose, callbacks=callbacks, validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        print()
        print("--------------------------------------------------------------------------------")
        print("================================================================================")
        print()
        print("Train has been successfully driven")
        print()
        
        return submodel_history, supmodel_history
    
    # Done
    def fit_submodel(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, callbacks=[EarlyStopping(verbose=1, restore_best_weights=True, patience=5)], validation_split=0.3, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False) : 
        """
        Fit submodels with input datasets.
        Denote that this method can cause time delay of kernel or overusage of RAM, depending the submodels you implant.

        Args : 
        - Essential args : 
            - x                 : train data
            - y                 : target data
        - Util args : 
            - epochs_sub        : maximum epoch that trains submodels
            - epochs_sup        : maximum epoch that trains supermodels
            - callbacks         : Predefined or custom callbacks to use when fitting the models
            - validation_split  : portion of validation datasets (0 ~ 1)

        Return : 
            - history_of_submodels ('list')

        You can use validation_data=(x_val, y_val) or F1ScoreCallback in "fit_submodel".
        """

        history_list = []
        print("Training submodels...")
        print()
        print(f"Number of submodels : {len(self.model_list)}")
        print(f"Submodels : {[self.model_list[i].name for i in range(len(self.model_list))]}")
        
        model_chekpoint = False
        for callback_check in callbacks : 
            if type(callback_check) is type(ModelCheckpoint('')) : 
                model_chekpoint = callback_check
                model_path = model_chekpoint.filepath
                filepath_sub = model_chekpoint.filepath + "/Submodel_checkpoint/" 

        for model in self.model_list : 
            print()
            print(f"Train {model.name}")
            
            if model_chekpoint : 
                model_chekpoint.filepath = filepath_sub + model.name + "_{epoch:02d}_{val_loss:.4f}.hdf5"
            
            history_list.append(model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing))
        
        model_chekpoint.filepath = model_path

        return history_list
    
    # Error : Can't use F1ScoreCallback
    # due to method overriding (self.predict : input data become incompatible shape)
    def fit_supermodel(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, callbacks=[EarlyStopping(verbose=1, restore_best_weights=True, patience=5)], validation_split=0.3, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False) : 
        """
        Fit supermodel with input datasets.
        -  If input datasets' shape is incompatible with shape=(num_submodels, num_data, num_classes), datasets will be automatically reshpaed to shape=(num_submodels, num_data, num_classes)
        
        For example : 
        - if sub_train_datasets             : x.shape = (6800, 28, 28, 1), y.shape = (6800)
        - input_datasetes will be reshaped  : x.shape = (num_submodels, 6800, num_classes), y.shape = (6800)

        Args : 
        - Essential args : 
            - x                 : train data
            - y                 : target data
        - Util args : 
            - epochs_sub        : maximum epoch that trains submodels
            - epochs_sup        : maximum epoch that trains supermodels
            - callbacks         : Predefined or custom callbacks to use when fitting the models
            - validation_split  : portion of validation datasets (0 ~ 1)
        
        Return : 
        - history_of_supermodel ('history' object)

        Note that you can't use validation_data=(x_val, y_val) or F1ScoreCallback due to method overriding.
        """
        
        print("Training supermodel...")
        print()
        print("Train model by prediction of submodels")
        print()
        train = self.predict_submodel(x)        # (model_num, data_num, 36)
        train_list = [data for data in train]
        
        print("Start training by prediction of submodels")

        for callback_check in callbacks : 
            if type(callback_check) is type(ModelCheckpoint('')) : 
                callback_check.filepath = callback_check.filepath + "/Supmodel_checkpoint/" + self.name + "_{epoch:02d}_{val_loss:.4f}.hdf5"

        model_history = self.__model__.fit(train_list, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split,  shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

        return model_history
    
    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        """
        Evaluate ensemble model with input datasets, using assigned metrics
        - Input datasets must be shaped as same as datasets that trained submodels.

        For example : 
        - if sub_train_datasets     : x.shape = (6800, 28, 28, 1), y.shape = (6800)
        - input_datasetes should    : x.shape = (6800, 28, 28, 1), y.shape = (6800)
        """

        print("Evaluating models' accuracy...")
        print()
        print("Evaluate model by comparing predictions and True values")
        print()
        model_pred = self.predict_submodel(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        
        pred_list = []
        for i in range(len(model_pred)) : 
            pred_list.append(model_pred[i])
        
        print("Start evaluating by comparing")
        return self.__model__.evaluate(pred_list, y, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
    
    # Done
    def predict(self, x, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        """
        Return prediction via ensemble model as np.array
        - Input datasets must be shaped as same as datasets that trained submodels.

        For example : 
        - if sub_train_datasets     : x.shape = (6800, 28, 28, 1)
        - input_datasetes should    : x.shape = (6800, 28, 28, 1)
        - return_array will be      : shape=(6800, num_classes)

        Return : 
        - prediction_via_ensemble_model (np.array)
        """

        print("================================================================================")
        print("--------------------------------------------------------------------------------")
        pred = self.predict_submodel(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        print("--------------------------------------------------------------------------------")
        pred = self.predict_supermodel(pred, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        print()
        print("--------------------------------------------------------------------------------")
        print("================================================================================")
        print()
        print("Prediction was successfully driven")
        print()
        return pred
    
    # Done
    def predict_submodel(self, x, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False) : 
        """
        Return prediction via submodels as np.array
        
        For example : 
        - if input_datasets         : x.shape = (6800, 28, 28, 1)
        - return_array will be      : x.shape = (num_submodels, 6800, num_classes)

        Return : 
        - prediction_via_submodels (np.array)
        """

        print("Making prediction by submodels...")
        print()
        print(f"Number of submodels : {len(self.model_list)}")
        print(f"Submodels : {[self.model_list[i].name for i in range(len(self.model_list))]}")
        print()
        pred_sub = []
        for model in self.model_list : 
            pred_sub.append(model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing))
        print()

        return np.array(pred_sub)       # (model_num, data_num, 36)
    
    # Done
    def predict_supermodel(self, x, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False) : 
        """
        Return prediction via supermodel as np.array
        -  If input datasets' shape is incompatible with shape=(num_submodels, num_data, num_classes), datasets will be automatically reshpaed to shape=(num_submodels, num_data, num_classes)
        
        For example : 
        - if sub_train_datasets             : x.shape = (6800, 28, 28, 1)
        - input_datasetes will be reshaped  : x.shape = (num_submodels, 6800, num_classes)
        - return_array will be              : shape=(6800, num_classes)

        Return : 
        - prediction_via_supermodel (np.array)
        """
        
        shape = x.shape
        if shape[0] != len(self.model_list) or shape[-1] != self.num_classes : 
            print(f"X is incompatible with shape=({len(self.model_list)}, {shape[1]}, {self.num_classes}), found shape={x.shape}")
            print(f"Changing X as shape=({len(self.model_list)}, {shape[1]}, {self.num_classes})...")
            print()
            x = self.predict_submodel(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
            print("Done")
            print()
        pred = []
        for i in range(len(x)) : 
            pred.append(x[i])
        
        print("Making prediction by supermodel...")
        return self.__model__.predict(pred, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
    
    # Done
    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        """
        Shows .summary() method (built-in method of keras.Model) of each submodels and supermodels
        """
        
        print(f"Number of submodels : {len(self.model_list)}")
        print(f"Submodels : {[self.model_list[i].name for i in range(len(self.model_list))]}")
        for model in self.model_list : 
            model.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable)
            print()
        self.__model__.summary()
        print()

# ====================================================================================================== #
# LeNet5
# Done

from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Layer

class LeNet5(Layer):
    def __init__(self, num_classes, name="LeNet5", *args, **kwargs):
        super(LeNet5, self).__init__(name=name, *args, **kwargs)
        
        self.conv1 = Conv2D(36, (5, 5), padding='valid', activation='relu', input_shape=(28, 28, 1))
        self.pool1 = AveragePooling2D(pool_size=(3, 3), strides=1)
        self.conv2 = Conv2D(72, (5, 5), padding='valid', activation='relu')
        self.pool2 = AveragePooling2D(pool_size=(3, 3), strides=1)
        self.flatten = Flatten()
        self.fc1 = Dense(num_classes * 3, activation='relu')
        self.fc2 = Dense(num_classes * 2, activation='relu')
        self.fc3 = Dense(num_classes, activation='softmax')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "conv1": self.conv1,
            "pool1": self.pool1,
            "conv2": self.conv2,
            "pool2": self.pool2,
            "flatten": self.flatten,
            "fc1": self.fc1,
            "fc2": self.fc2,
            "fc3": self.fc3
        })
        return config

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# ====================================================================================================== #
# GoogLeNet
# Done

from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, Concatenate, Layer

class InceptionModule(Layer):
    def __init__(self, c1_1, c1_2, c3, c1_3, c5, c1_4, **kwargs) :
        super(InceptionModule, self).__init__(**kwargs)
        
        self.conv1_1 = Conv2D(c1_1, (1, 1), activation='relu', padding='same')
        self.conv1_2 = Conv2D(c1_2, (1, 1), activation='relu', padding='same')
        self.conv1_3 = Conv2D(c1_3, (1, 1), activation='relu', padding='same')
        self.conv1_4 = Conv2D(c1_4, (1, 1), activation='relu', padding='same')
        self.conv3 = Conv2D(c3, (3, 3), activation='relu', padding='same')
        self.conv5 = Conv2D(c5, (5, 5), activation='relu', padding='same')
        self.max_pool = MaxPooling2D((3, 3), strides=1, padding='same')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "conv1_1": self.conv1_1,
            "conv1_2": self.conv1_2,
            "conv1_3": self.conv1_3,
            "conv1_4": self.conv1_4,
            "conv3": self.conv1_3,
            "conv5": self.conv5,
            "max_pool": self.max_pool,
        })
        return config

    def call(self, inputs):
        x1 = self.conv1_1(inputs)
        x3 = self.conv1_2(inputs)
        x5 = self.conv1_3(inputs)
        x_pool = self.max_pool(inputs)

        x3 = self.conv3(x3)
        x5 = self.conv5(x5)
        x_pool = self.conv1_4(x_pool)
        
        x = Concatenate(axis=-1)([x1, x3, x5, x_pool])
        
        return x

class AuxiliaryClassifier(Layer) : 
    def __init__(self, num_classes, **kwargs) :
        super(AuxiliaryClassifier, self).__init__(**kwargs)

        self.avg_pool = AveragePooling2D((5, 5), strides=1, padding='valid')
        self.conv1 = Conv2D(128, (1, 1), activation='relu', padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(num_classes * 3, activation='relu')
        self.drop = Dropout(0.7)
        self.fc2 = Dense(num_classes, activation='softmax')

    def get_config(self):
        config = super().get_config()
        config.update({
            "avg_pool": self.avg_pool,
            "conv1": self.conv1,
            "flatten": self.flatten,
            "fc1": self.fc1,
            "drop": self.drop,
            "fc2": self.fc2,
        })
        return config

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x

class GoogLeNet(Layer):
    def __init__(self, num_classes=10, name="GoogLeNet", *args, **kwargs) :
        super(GoogLeNet, self).__init__(name=name, *args, **kwargs)
        
        self.conv1 = Conv2D(64, (7, 7), strides=1, activation='relu', padding='same')
        self.max_pool1 = MaxPooling2D((3, 3), strides=1, padding='same')
        self.conv2 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='valid')
        self.conv3 = Conv2D(192, (3, 3), activation='relu', padding='same')
        self.max_pool2 = MaxPooling2D((3, 3), strides=1, padding='same')
        
        self.inception_3a = InceptionModule(64, 96, 128, 16, 32, 32)        # 64, 96, 128, 16, 32, 32
        self.inception_3b = InceptionModule(128, 128, 192, 32, 96, 64)      # 128, 128, 192, 32, 96, 64
        self.max_pool3 = MaxPooling2D((3, 3), strides=1, padding='same')

        self.inception_4a = InceptionModule(192, 96, 208, 16, 48, 64)       # 192, 96, 208, 16, 48, 64

        self.aux_1 = AuxiliaryClassifier(num_classes)
        self.inception_4b = InceptionModule(160, 112, 224, 24, 64, 64)      # 160, 112, 224, 24, 64, 64

        self.inception_4c = InceptionModule(128, 128, 256, 24, 64, 64)      # 128, 128, 256, 24, 64, 64
        self.inception_4d = InceptionModule(112, 144, 288, 32, 64, 64)      # 112, 144, 288, 32, 64, 64

        self.aux_2 = AuxiliaryClassifier(num_classes)
        self.inception_4e = InceptionModule(256, 160, 320, 32, 128, 128)    # 256, 160, 320, 32, 128, 128
        self.max_pool4 = MaxPooling2D((3, 3), strides=1, padding='same')

        self.inception_5a = InceptionModule(256, 160, 320, 32, 128, 128)    # 256, 160, 320, 32, 128, 128
        self.inception_5b = InceptionModule(384, 192, 384, 48, 128, 128)    # 384, 192, 384, 48, 128, 128

        self.avg_pool = GlobalAveragePooling2D()
        self.dropout = Dropout(0.4)
        self.fc = Dense(num_classes, activation='softmax')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "conv1": self.conv1,
            "max_pool1": self.max_pool1,
            "conv2": self.conv2,
            "conv3": self.conv3,
            "max_pool2": self.max_pool2,
            "inception_3a": self.inception_3a,
            "inception_3b": self.inception_3b,
            "max_pool3": self.max_pool3,
            "inception_4a": self.inception_4a,
            "aux_1": self.aux_1,
            "inception_4b": self.inception_4b,
            "inception_4c": self.inception_4c,
            "inception_4d": self.inception_4d,
            "aux_2": self.aux_2,
            "inception_4e": self.inception_4e,
            "max_pool4": self.max_pool4,
            "inception_5a": self.inception_5a,
            "inception_5b": self.inception_5b,
            "avg_pool": self.avg_pool,
            "dropout": self.dropout,
            "fc": self.fc
        })
        return config

    def call(self, input_layer):
        x1 = self.conv1(input_layer)
        x1 = self.max_pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.max_pool2(x1)

        x1 = self.inception_3a(x1)
        x1 = self.inception_3b(x1)
        x1 = self.max_pool3(x1)

        x1 = self.inception_4a(x1)
        x2 = self.aux_1(x1)
        x1 = self.inception_4b(x1)
        x1 = self.inception_4c(x1)
        x1 = self.inception_4d(x1)
        x3 = self.aux_2(x1)
        x1 = self.inception_4e(x1)
        x1 = self.max_pool4(x1)

        x1 = self.inception_5a(x1)
        x1 = self.inception_5b(x1)

        x1 = self.avg_pool(x1)
        x1 = self.fc(x1)
        
        return [x1, x2, x3]

# ====================================================================================================== #
# VGG16
# Done

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Layer

class VGG16(Layer):
    def __init__(self, num_classes=10, **kwargs):
        super(VGG16, self).__init__(**kwargs)
        
        self.conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.max_pool1 = AveragePooling2D((3, 3), strides=2)
        
        self.conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.max_pool2 = AveragePooling2D((3, 3), strides=1)
        
        self.conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.max_pool3 = AveragePooling2D((3, 3), strides=2)
        
        self.conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv4_3 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.max_pool4 = AveragePooling2D((3, 3), strides=1)
        
        self.conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.max_pool5 = AveragePooling2D((3, 3), strides=1)
        
        self.flatten = Flatten()
        self.fc1 = Dense(num_classes * 6, activation='relu')
        self.fc2 = Dense(num_classes * 3, activation='relu')
        self.fc3 = Dense(num_classes, activation='softmax')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "conv1_1": self.conv1_1,
            "conv1_2": self.conv1_2,
            "max_pool1": self.max_pool1,

            "conv2_1": self.conv2_1,
            "conv2_2": self.conv2_2,
            "max_pool2": self.max_pool2,
            
            "conv3_1": self.conv3_1,
            "conv3_2": self.conv3_2,
            "conv3_3": self.conv3_3,
            "max_pool3": self.max_pool3,
            
            "conv4_1": self.conv4_1,
            "conv4_2": self.conv4_2,
            "conv4_3": self.conv4_3,
            "max_pool4": self.max_pool4,

            "conv5_1": self.conv5_1,
            "conv5_2": self.conv5_2,
            "conv5_3": self.conv5_3,
            "max_pool5": self.max_pool5,

            "flatten": self.flatten,
            "fc1": self.fc1,
            "fc2": self.fc2,
            "fc3": self.fc3
        })
        return config

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.max_pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max_pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.max_pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.max_pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.max_pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# ====================================================================================================== #