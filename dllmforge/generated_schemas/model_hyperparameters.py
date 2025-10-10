"""
Generated Pydantic schema for information extraction
"""
from pydantic import BaseModel, Field, conint, confloat, constr
from typing import Optional, List

class ModelArchitecture(BaseModel):
    model_type: constr(strip_whitespace=True, min_length=1) = Field(..., description="Type of the model architecture, e.g., CNN, RNN, Transformer.")
    layers: Optional[List[constr(strip_whitespace=True, min_length=1)]] = Field(None, description="List of layers used in the model architecture, e.g., ['Conv2D', 'MaxPooling', 'Dense'].")
    neurons_per_layer: Optional[List[conint(gt=0)]] = Field(None, description="Number of neurons in each layer, corresponding to the layers list.")

class TrainingParameters(BaseModel):
    learning_rate: confloat(gt=0) = Field(..., description="Learning rate used for training the model.")
    batch_size: conint(gt=0) = Field(..., description="Batch size used during training.")
    epochs: conint(gt=0) = Field(..., description="Number of epochs for training the model.")

class OptimizationSettings(BaseModel):
    optimizer: constr(strip_whitespace=True, min_length=1) = Field(..., description="Optimizer used for training, e.g., 'Adam', 'SGD'.")
    loss_function: constr(strip_whitespace=True, min_length=1) = Field(..., description="Loss function used for training, e.g., 'categorical_crossentropy', 'mean_squared_error'.")

class RegularizationTechniques(BaseModel):
    dropout_rate: Optional[confloat(ge=0, le=1)] = Field(None, description="Dropout rate used for regularization, if applicable.")
    other_techniques: Optional[List[constr(strip_whitespace=True, min_length=1)]] = Field(None, description="Other regularization techniques used, e.g., ['L2 regularization', 'Batch normalization'].")

class ModelHyperparameters(BaseModel):
    architecture: ModelArchitecture = Field(..., description="Details of the model architecture.")
    training_parameters: TrainingParameters = Field(..., description="Parameters related to the training process.")
    optimization_settings: OptimizationSettings = Field(..., description="Settings related to optimization during training.")
    regularization_techniques: Optional[RegularizationTechniques] = Field(None, description="Regularization techniques applied to the model.")
