"""
Generated Pydantic schema for information extraction
"""
from pydantic import BaseModel, Field, conint, confloat, constr
from typing import Optional, List

class ModelArchitecture(BaseModel):
    model_type: str = Field(..., description="Type of the model architecture, e.g., CNN, RNN, etc.")
    layers: List[str] = Field(..., description="List of layers used in the model, e.g., ['Conv2D', 'MaxPooling', 'Dense'].")
    neurons_per_layer: List[conint(ge=1)] = Field(..., description="Number of neurons in each layer, must be a positive integer.")

class TrainingParameters(BaseModel):
    learning_rate: confloat(gt=0) = Field(..., description="Learning rate used for training, must be a positive float.")
    batch_size: conint(ge=1) = Field(..., description="Batch size used during training, must be a positive integer.")
    epochs: conint(ge=1) = Field(..., description="Number of epochs for training, must be a positive integer.")

class OptimizationSettings(BaseModel):
    optimizer: str = Field(..., description="Optimizer used for training, e.g., 'Adam', 'SGD'.")
    loss_function: str = Field(..., description="Loss function used, e.g., 'categorical_crossentropy', 'mean_squared_error'.")

class RegularizationTechniques(BaseModel):
    dropout_rate: Optional[confloat(ge=0, le=1)] = Field(None, description="Dropout rate used for regularization, must be between 0 and 1.")
    other_techniques: Optional[List[str]] = Field(None, description="List of other regularization techniques used, e.g., ['L2', 'BatchNormalization'].")

class ModelHyperparameters(BaseModel):
    architecture: ModelArchitecture = Field(..., description="Details of the model architecture.")
    training_parameters: TrainingParameters = Field(..., description="Parameters used for training the model.")
    optimization_settings: OptimizationSettings = Field(..., description="Settings related to optimization during training.")
    regularization_techniques: Optional[RegularizationTechniques] = Field(None, description="Regularization techniques applied to the model.")
