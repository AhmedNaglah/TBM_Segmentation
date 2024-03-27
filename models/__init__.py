import importlib
from models.base import Base

def create_model_by_name(model_name, config):

    model_filename = "models." + model_name 
    classes = importlib.import_module(model_filename)
    found = False
    model = None
    for name, cls in classes.__dict__.items():
        if name == model_name and issubclass(cls, Base):
            model = cls(config)
            found = True
    return model, found

    