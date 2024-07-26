from importlib import import_module
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Given a package module for a model cls and configreturn a model instance
    """

    def __init__(self):
        pass

    @classmethod
    def get_instance_from_cofing(
        cls, package_module_cls: str, config: dict = {}
    ):
        """
        Get the model's package module cls
        Create an instance using the config
        """
        model_cls_module, model_cls = package_module_cls.split(":")
        logger.info(f"Creating model: {model_cls_module} - {model_cls}")

        model_cls_obj = getattr(import_module(model_cls_module), model_cls)
        logger.info(f"Creating {model_cls} using {config}")
        return model_cls_obj(**config)
