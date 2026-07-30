from .deev_censor import DeevCensorError, DeevGenitalAnusCensor

NODE_CLASS_MAPPINGS = {
    "DeevGenitalAnusCensor": DeevGenitalAnusCensor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeevGenitalAnusCensor": "Deev Genital/Anus Censor (01miku)",
}

__all__ = [
    "DeevCensorError",
    "DeevGenitalAnusCensor",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
