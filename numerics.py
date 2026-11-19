"""
Floating-point mode of convolutions and matrix products.

On Ampere and newer NVIDIA GPUs PyTorch runs cuDNN convolutions in TF32 by default: operands are
rounded to 10 mantissa bits, so bit-flips in the low FP32 mantissa bits of a weight are masked
and results depend on the GPU model. SpikingJET therefore runs in exact FP32 by default; TF32 is
opt-in (--tf32) for speed when bit-level FP32 sensitivity does not matter. The mode is the same
on every GPU and on the CPU (where TF32 does not exist).
"""
import torch


def set_tf32(enabled: bool = False, deterministic: bool = True) -> dict:
    """Allow (enabled=True) or forbid TF32 in cuDNN convolutions and CUDA matmuls; with
    deterministic, also pick deterministic cuDNN algorithms. Returns the resulting mode."""
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return mode()


def mode() -> dict:
    return {"tf32_conv": bool(torch.backends.cudnn.allow_tf32),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)}


def add_argument(parser):
    parser.add_argument("--tf32", action="store_true",
                        help="allow TF32 in convolutions/matmuls on Ampere+ GPUs (faster; masks bit-flips "
                             "in FP32 mantissa bits 0-12 of conv operands and makes results GPU dependent). "
                             "Default: exact FP32.")
