from typing import Dict
import torch
from torch.nn import Module
from dataclasses import dataclass


class InferenceServer:
  ''' Handles the execution (inference) of all models.
  '''

  @dataclass
  class Model:
    model: Module



  MODELS: Dict[str, Model] = {}


  @staticmethod
  def addModel(name: str, model: Module) -> None:
    if InferenceServer.hasModel(name):
      raise Exception(f"There is already a model named \"{name}\" in the InferenceServer")
    InferenceServer.MODELS[name] = InferenceServer.Model(model=model)
  

  @staticmethod
  def hasModel(name: str) -> bool:
    return name in InferenceServer.MODELS.keys()
  
  @staticmethod
  def removeModel(name: str):
    if name in InferenceServer.MODELS.keys():
      del InferenceServer.MODELS[name]

  @staticmethod
  def model(name: str) -> Module:
    return InferenceServer.MODELS[name].model


  @staticmethod
  def inference(model_name: str, unpack: bool, *args, **kwargs):
    if not InferenceServer.hasModel(model_name):
      raise Exception(f"No model called \"{model_name}\" has been loaded in the InferenceServer")
    model_obj = InferenceServer.MODELS[model_name]
    if unpack:
      args = [a.unsqueeze(0) for a in args]
      kwargs = {k:v.unsqueeze(0) for k,v in kwargs.items()}
    result = model_obj.model(*args, **kwargs)
    if unpack:
      if isinstance(result, tuple):
        return tuple(r.squeeze(0) for r in result)
      return result.squeeze(0)
    return result