from tkdet.utils.registry import Registry

__all__ = ["RPN_REGISTRY", "build_rpn"]

RPN_REGISTRY = Registry("RPN")


def build_rpn(cfg, input_shape):
    name = cfg.MODEL.RPN.NAME
    if name == "PrecomputedProposals":
        return None

    return RPN_REGISTRY.get(name)(cfg, input_shape)
