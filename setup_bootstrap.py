"""
sd-forge-lora-trainer / setup_bootstrap.py
==========================================

Standalone install script, run by setup_venv.bat (double-click) or by Forge's own setup (tools\\initialize.ps1),
with a Python 3.10 / 3.11 interpreter (kohya-ss/sd-scripts is tested on those).

Does the heavy one-time setup the LoRA Trainer tab needs:
  1. Clone kohya-ss/sd-scripts into ./sd-scripts/ at the commit this extension was tested with (if not there)
  2. Create a sandboxed venv at ./sd-scripts-venv/ using THIS interpreter
     (so kohya's torch does not clobber Forge's torch 2.10.0+cu130).
  3. pip-install PyTorch for the graphics card: 2.8.0 + CUDA 12.8 on an RTX 50 (Blackwell, compute capability 10+,
     as sd-scripts' README says), else 2.6.0 + CUDA 12.4 (the version sd-scripts is tested with).
  4. pip-install kohya's own requirements.txt (most reliable version pins).
  5. pip-install onnxruntime-gpu (for WD14 tagger; not in kohya's reqs).
  6. Check: a calculation on the GPU, and the training script's imports.

Idempotent: when the venv already passes the check (with the right PyTorch for the card), it says so and stops;
otherwise it repairs what is missing - a PyTorch build that cannot run on the card is replaced.

Exit code 0 = ready. LORA_TRAINER_TORCH_INDEX / LORA_TRAINER_TORCH override the PyTorch index / version pins.

This file deliberately has NO imports of gradio / modules / backend so it
can be invoked outside the WebUI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SD_SCRIPTS_DIR = ROOT / "sd-scripts"
SD_SCRIPTS_VENV = ROOT / "sd-scripts-venv"

SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
SD_SCRIPTS_COMMIT = "068bcd7"          # main as of the merge of PR #2359 - the version this extension was tested with

TORCH_BLACKWELL = ("https://download.pytorch.org/whl/cu128", ["torch==2.8.0", "torchvision==0.23.0"])
TORCH_DEFAULT = ("https://download.pytorch.org/whl/cu124", ["torch==2.6.0", "torchvision==0.21.0"])

EXTRA_PACKAGES = ["onnxruntime-gpu", "numpy<2.0"]


def venv_python() -> Path:
    if os.name == "nt":
        return SD_SCRIPTS_VENV / "Scripts" / "python.exe"
    return SD_SCRIPTS_VENV / "bin" / "python"


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if rc != 0:
        print(f"\n!! command failed with exit code {rc}", flush=True)
        sys.exit(rc)


def gpu() -> tuple[int, int]:
    """(compute capability major, driver major) of the first NVIDIA card; (0, 0) without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        cc, drv = [x.strip() for x in out[0].split(",")]
        return int(cc.split(".")[0]), int(drv.split(".")[0])
    except Exception:
        return 0, 0


def torch_plan() -> tuple[str, list[str]]:
    cc, drv = gpu()
    index, pkgs = TORCH_BLACKWELL if cc >= 10 else TORCH_DEFAULT
    if cc >= 10 and drv and drv < 570:
        print(f"!! an RTX 50 card needs NVIDIA driver 570 or newer for PyTorch 2.8 / CUDA 12.8 (found {drv})", flush=True)
        sys.exit(1)
    index = os.environ.get("LORA_TRAINER_TORCH_INDEX", index)
    pkgs = os.environ.get("LORA_TRAINER_TORCH", " ".join(pkgs)).split()
    return index, pkgs


CHECK = (
    "import sys, torch; "
    "want = sys.argv[1]; "
    "assert torch.__version__.split('+')[0] == want, f'torch {torch.__version__}, want {want}'; "
    "x = torch.ones(8, device='cuda'); assert float((x * 2).sum()) == 16; "
    "import accelerate, diffusers, transformers, safetensors; "
    "print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
)


def ready(torch_version: str) -> bool:
    """The venv runs a calculation on the GPU with the expected PyTorch and has the training stack."""
    vp = venv_python()
    if not vp.exists() or not (SD_SCRIPTS_DIR / "sdxl_train_network.py").exists():
        return False
    r = subprocess.run([str(vp), "-c", CHECK, torch_version], capture_output=True, text=True, cwd=str(SD_SCRIPTS_DIR))
    if r.returncode == 0:
        print("      " + r.stdout.strip(), flush=True)
        return True
    err = (r.stderr.strip().splitlines() or ["?"])[-1]
    print(f"      not ready: {err}", flush=True)
    return False


def step_1_clone():
    train_script = SD_SCRIPTS_DIR / "sdxl_train_network.py"
    if train_script.exists():
        print(f"[1/6] sd-scripts already cloned at {SD_SCRIPTS_DIR}, skipping.", flush=True)
        return
    print(f"[1/6] cloning {SD_SCRIPTS_REPO} at {SD_SCRIPTS_COMMIT} ...", flush=True)
    if SD_SCRIPTS_DIR.exists():
        shutil.rmtree(SD_SCRIPTS_DIR, ignore_errors=True)
    run(["git", "clone", "--quiet", SD_SCRIPTS_REPO, str(SD_SCRIPTS_DIR)])
    run(["git", "-C", str(SD_SCRIPTS_DIR), "-c", "advice.detachedHead=false", "checkout", "--quiet", SD_SCRIPTS_COMMIT])


def step_2_venv():
    vp = venv_python()
    if vp.exists():
        print(f"[2/6] venv already exists at {SD_SCRIPTS_VENV}, skipping.", flush=True)
        return
    print(f"[2/6] creating sandboxed venv at {SD_SCRIPTS_VENV} ...", flush=True)
    print(f"      using interpreter: {sys.executable}", flush=True)
    print(f"      python version   : {sys.version.split()[0]}", flush=True)

    # Try the stdlib venv module first. Some stripped/uv-distributed Python
    # installs have NO stdlib venv module but DO have a working virtualenv
    # package on the path - fall back to that.
    cmd_venv = [sys.executable, "-m", "venv", str(SD_SCRIPTS_VENV)]
    cmd_vex = [sys.executable, "-m", "virtualenv", str(SD_SCRIPTS_VENV)]

    print("\n$ " + " ".join(str(c) for c in cmd_venv), flush=True)
    rc = subprocess.call([str(c) for c in cmd_venv])
    if rc != 0 or not vp.exists():
        print("   stdlib venv failed (rc={}); trying virtualenv ...".format(rc), flush=True)
        print("\n$ " + " ".join(str(c) for c in cmd_vex), flush=True)
        rc = subprocess.call([str(c) for c in cmd_vex])
        if rc != 0:
            print(f"\n!! virtualenv also failed (rc={rc})", flush=True)
            sys.exit(rc)
    if not vp.exists():
        print(f"!! venv python missing at {vp}", flush=True)
        sys.exit(1)


def step_3_pip_base():
    print("[3/6] upgrading pip / setuptools / wheel in sandboxed venv ...", flush=True)
    run([str(venv_python()), "-m", "pip", "install", "--upgrade", "--quiet", "pip", "setuptools", "wheel"])


def step_4_pip_torch_and_requirements(index: str, pkgs: list[str]):
    print(f"[4a/6] installing {' '.join(pkgs)} from {index} ...", flush=True)
    # an older torch already in the venv (2.6.0 + cu124 imports on an RTX 50 but cannot run there) is replaced by the pin
    run([str(venv_python()), "-m", "pip", "install", "--index-url", index, *pkgs])

    reqs = SD_SCRIPTS_DIR / "requirements.txt"
    if not reqs.exists():
        print(f"!! kohya requirements.txt missing at {reqs}", flush=True)
        sys.exit(1)

    print("[4b/6] installing kohya's requirements.txt ...", flush=True)
    run([str(venv_python()), "-m", "pip", "install", "--use-pep517", "-r", str(reqs)], cwd=SD_SCRIPTS_DIR)


def step_5_extras():
    print("[5/6] installing extras (WD14 tagger deps) ...", flush=True)
    run([str(venv_python()), "-m", "pip", "install", *EXTRA_PACKAGES])


def main():
    index, pkgs = torch_plan()
    torch_version = next((p.split("==")[1] for p in pkgs if p.startswith("torch==")), "")
    print("=" * 70, flush=True)
    print("sd-forge-lora-trainer / bootstrap", flush=True)
    print("=" * 70, flush=True)
    print(f"  install root   : {ROOT}", flush=True)
    print(f"  sd-scripts     : {SD_SCRIPTS_DIR} (at {SD_SCRIPTS_COMMIT})", flush=True)
    print(f"  sandboxed venv : {SD_SCRIPTS_VENV}", flush=True)
    print(f"  torch          : {' '.join(pkgs)} from {index}", flush=True)
    print(f"  invoker python : {sys.executable} ({sys.version.split()[0]})", flush=True)
    if sys.version_info[:2] not in [(3, 10), (3, 11)]:
        v = "{}.{}".format(sys.version_info[0], sys.version_info[1])
        print(f"  WARNING: kohya recommends Python 3.10; running on {v}", flush=True)
        print("  Some deps may lack wheels for newer Python.", flush=True)
    print("=" * 70, flush=True)

    print("[check] is it set up already?", flush=True)
    if torch_version and ready(torch_version):
        print("  READY - nothing to do.", flush=True)
        return
    print("  expected runtime: 5-15 minutes, about 8 GB (depends on bandwidth)", flush=True)

    step_1_clone()
    step_2_venv()
    step_3_pip_base()
    step_4_pip_torch_and_requirements(index, pkgs)
    step_5_extras()

    print("[6/6] checking ...", flush=True)
    if torch_version and not ready(torch_version):
        print("!! the training environment does not work on this graphics card - see the message above", flush=True)
        sys.exit(1)

    print()
    print("=" * 70, flush=True)
    print("  DONE.", flush=True)
    print("  The LoRA Trainer is ready. Open Forge and look for the", flush=True)
    print("  'LoRA Trainer' tab next to txt2img / img2img.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n!! interrupted by user", flush=True)
        sys.exit(130)
