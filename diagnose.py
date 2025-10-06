import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.hooks import RemovableHandle

from experiment import create_model_dict_path
from helpers.arguments import parse_arguments
from helpers.datasets import DataSet
from helpers.split_data import split_data_per_fold
from helpers.utils import coo_to_csr
from models.gfm import GFM, GFMArgs


def build_model(args) -> GFM:
    gfm_args = GFMArgs(
        hid_channel=args.hid_channel,
        num_layers=args.num_layers,
        ls_num_layers=args.ls_num_layers,
        gnn_type=args.gnn_type,
        lp_ratio=args.lp_ratio,
    )
    return GFM(gfm_args)


def resolve_checkpoint(args) -> Path:
    checkpoint_dir = Path(create_model_dict_path(args))
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    candidates = sorted(checkpoint_dir.glob("Seed*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files matching 'Seed*.pt' found in {checkpoint_dir}"
        )
    return candidates[0]


def load_cornell_data(args, seed: int = 0):
    data = DataSet.cornell.load()
    data = split_data_per_fold(seed=seed, data=data, ls_num_layers=args.ls_num_layers,
                               dataset_name=DataSet.cornell.name)

    if args.gnn_type.uses_triton():
        rowptr, indices = coo_to_csr(data.edge_index[0], data.edge_index[1], num_nodes=data.x.shape[0])
        setattr(data, "rowptr", rowptr)
        setattr(data, "indices", indices)
        setattr(data, "edge_index", [])
    else:
        setattr(data, "rowptr", [])
        setattr(data, "indices", [])
    return data


def print_parameter_report(model: GFM) -> None:
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        num_parameters = parameter.numel()
        total_params += num_parameters
        print(f"{name}: shape={tuple(parameter.shape)} params={num_parameters}")
    print(f"Total learnable parameters: {total_params}")


def determine_device(args) -> torch.device:
    if args.gpu is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_shape_hooks(model: GFM) -> List[RemovableHandle]:
    handles: List[RemovableHandle] = []

    for idx, layer in enumerate(model.equiv_layers):
        def make_hook(layer_idx: int):
            def hook(module, inputs, outputs):
                out_x, out_y = outputs
                print(
                    f"EquivLayer {layer_idx}: x shape={tuple(out_x.shape)}, y shape={tuple(out_y.shape)}"
                )
            return hook

        handles.append(layer.register_forward_hook(make_hook(idx)))
    return handles


def perform_inference(model: GFM, data, device: torch.device, args) -> None:
    model = model.to(device)
    model.eval()

    hooks = register_shape_hooks(model)

    with torch.no_grad():
        train_y = data.y_mat.clone()
        train_y[~data.train_mask] = 0

        edge_index = None if args.gnn_type.uses_triton() else data.edge_index
        outputs, _ = model(
            data.x,
            train_y=train_y,
            xy_conversions=data.xy_conversions,
            is_batch=False,
            device=device,
            edge_index=edge_index,
            rowptr=data.rowptr,
            indices=data.indices,
        )

    for handle in hooks:
        handle.remove()

    print(f"Final output logits shape: {tuple(outputs.shape)}")


if __name__ == "__main__":
    args = parse_arguments()
    try:
        checkpoint_path = resolve_checkpoint(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model = build_model(args)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        print(f"Failed to load state dict: {exc}", file=sys.stderr)
        sys.exit(1)

    print(model)
    print_parameter_report(model)

    device = determine_device(args)
    data = load_cornell_data(args=args)

    print(f"Cornell feature matrix shape: {tuple(data.x.shape)}")
    print(f"Cornell one-hot label matrix shape: {tuple(data.y_mat.shape)}")
    print(f"Cornell label vector shape: {tuple(data.y.shape)}")

    perform_inference(model=model, data=data, device=device, args=args)