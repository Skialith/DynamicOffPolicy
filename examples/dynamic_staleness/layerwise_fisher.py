"""Exact J^T F_z Jv without a whole-model second-order graph.

For an eval Qwen3 model outside FSDP, propagate parameter/input tangents one
module at a time. Retain detached inputs on CPU or their layer GPU, then
recompute each module for an ordinary VJP and release its input. This is the
frozen-anchor KL Hessian at the anchor, not a Hessian of an arbitrary loss or of KL away from its anchor.
"""

from contextlib import nullcontext

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    create_causal_mask, create_sliding_window_causal_mask,
)


def module_jvp(module, value, tangent, directions):
    named = dict(module.named_parameters())
    vectors = {name: directions[id(parameter)] for name, parameter in named.items()}
    # No reverse-mode graph: forward AD still propagates parameter tangents.
    with torch.no_grad():
        def call(parameters, inputs):
            return torch.func.functional_call(module, parameters, (inputs,))

        if tangent is None:  # Integer embedding indices are not differentiable.
            primal, result = torch.func.jvp(lambda parameters: call(parameters, value),
                                          (named,), (vectors,))
        else:
            primal, result = torch.func.jvp(call, (named, value), (vectors, tangent))
    return primal.detach(), result.detach()


def _layerwise_derivative(model, row, anchor, direction, saved_context=None, progress=None,
                          layer_inputs_cpu_offload=True, accumulator=None):
    if model.training or model.config.model_type != "qwen3":
        raise ValueError("Layerwise Fisher requires an eval, unwrapped Qwen3 model")
    parameters = list(model.parameters())
    if direction is not None and len(parameters) != len(direction):
        raise ValueError("Parameter direction length mismatch")
    directions = {} if direction is None else {
        id(parameter): vector for parameter, vector in zip(parameters, direction)}
    indices_by_id = {id(parameter): index for index, parameter in enumerate(parameters)}
    covered = set()
    tape = []
    if saved_context is None:
        saved_context = nullcontext

    def emit(stage):
        if progress is not None:
            progress(stage)

    def forward(module, value, tangent, label):
        device = next(module.parameters()).device
        value = value.to(device)
        if tangent is not None:
            tangent = tangent.to(device)
        saved_input = value.detach()
        if layer_inputs_cpu_offload:
            saved_input = saved_input.cpu()
        tape.append((module, saved_input, label))
        covered.update(id(parameter) for parameter in module.parameters())
        if direction is None:
            with torch.no_grad():
                primal, result = module(value).detach(), None
        else:
            primal, result = module_jvp(module, value, tangent, directions)
        emit(f"primal_{label}" if direction is None else f"jvp_{label}")
        return primal, result

    hidden, tangent = forward(model.model.embed_tokens, row["input_ids"], None, "embedding")
    position_ids = row["position_ids"].to(hidden.device)
    cache_position = torch.arange(hidden.shape[1], device=hidden.device)
    mask_kwargs = dict(config=model.config, input_embeds=hidden, attention_mask=None,
                       cache_position=cache_position, past_key_values=None, position_ids=position_ids)
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if model.model.has_sliding_layers:
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    with torch.no_grad():
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

    # Wrap fixed positional/mask arguments without changing a layer's parameters.
    class Decoder(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, value):
            return self.layer(value, attention_mask=masks[self.layer.attention_type],
                              position_ids=position_ids, past_key_values=None, use_cache=False,
                              cache_position=cache_position, position_embeddings=position_embeddings)

    for index, layer in enumerate(model.model.layers):
        hidden, tangent = forward(Decoder(layer), hidden, tangent, f"layer_{index}")
    hidden, tangent = forward(model.model.norm, hidden, tangent, "norm")
    selected = row["indices"].to(hidden.device)
    full_hidden_shape = hidden.shape
    hidden = hidden[:, selected, :]
    if tangent is not None:
        tangent = tangent[:, selected, :]
    logits, tangent = forward(model.lm_head, hidden, tangent, "head")
    if covered != set(indices_by_id):
        raise ValueError("Layerwise Fisher did not cover every model parameter")
    # F_z r, including the unchanged prompt/position aggregation weights.
    with torch.no_grad():
        probabilities = anchor.to(logits.device).exp()
        if direction is None:
            covector = logits[0].double().softmax(-1) * probabilities.sum(-1, keepdim=True) - probabilities
        else:
            r = tangent[0].double()
            covector = probabilities * (r - (probabilities * r).sum(-1, keepdim=True))
            del r
        covector = (covector * row["weights"].to(logits.device)[:, None]).to(logits.dtype).unsqueeze(0)
    del logits, tangent, hidden, probabilities
    emit("fisher_logits_done")
    result = accumulator if accumulator is not None else [torch.zeros_like(parameter) for parameter in parameters]
    while tape:
        module, saved_input, label = tape.pop()
        local_parameters = list(module.parameters())
        device = local_parameters[0].device
        value = saved_input.to(device)
        differentiable_input = value.is_floating_point()
        if differentiable_input:
            value.requires_grad_(True)
        inputs = [value, *local_parameters] if differentiable_input else local_parameters
        with saved_context():
            output = module(value)
            gradients = torch.autograd.grad(output, inputs, grad_outputs=covector.to(output))
        if differentiable_input:
            covector, *parameter_gradients = gradients
        else:
            parameter_gradients = gradients
        for parameter, gradient in zip(local_parameters, parameter_gradients):
            result[indices_by_id[id(parameter)]].add_(gradient)
        del output, inputs, gradients, parameter_gradients, value, saved_input
        if label == "head":
            # Transpose of the selected-position gather before the output head.
            full = covector.new_zeros(full_hidden_shape)
            full.index_add_(1, selected.to(full.device), covector)
            covector = full
        emit(f"vjp_{label}")
    emit("row_derivatives_done")
    return result


def layerwise_fisher_product(model, row, anchor, direction, saved_context=None, progress=None,
                             layer_inputs_cpu_offload=True, accumulator=None):
    """Add a row FVP into accumulator in place, or allocate one when omitted."""
    return _layerwise_derivative(model, row, anchor, direction, saved_context, progress,
                                 layer_inputs_cpu_offload, accumulator)


def layerwise_kl_gradient(model, row, anchor, saved_context=None, layer_inputs_cpu_offload=True):
    """Frozen-KL first derivative, with the same bounded-memory VJP path."""
    return _layerwise_derivative(model, row, anchor, None, saved_context,
                                 layer_inputs_cpu_offload=layer_inputs_cpu_offload)
