from __future__ import annotations


def validate_model_class_alignment(model, class_names):
    """Raise clear error if model output size doesn't match class mapping."""
    out_shape = getattr(model, "output_shape", None)
    if not out_shape or not isinstance(out_shape, tuple) or len(out_shape) < 2:
        return

    num_outputs = out_shape[-1]
    if num_outputs is None:
        return

    if int(num_outputs) != len(class_names):
        raise ValueError(
            "Model/class mismatch: model outputs "
            f"{num_outputs} classes but class names list has {len(class_names)} entries."
        )
