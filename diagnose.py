import sys
from pathlib import Path

import torch

from experiment import create_model_dict_path
from helpers.arguments import parse_arguments
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


def print_parameter_report(model: GFM) -> None:
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        num_parameters = parameter.numel()
        total_params += num_parameters
        print(f"{name}: shape={tuple(parameter.shape)} params={num_parameters}")
    print(f"Total learnable parameters: {total_params}")


def main() -> None:
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


if __name__ == "__main__":
    main()