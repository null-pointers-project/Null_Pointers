"""
Smart Story Teller - Device & Environment Setup Checker
Run this script to verify your environment is correctly configured.
Usage:
    - Mac (training): python scripts/check_environment.py --mode train
    - Windows (deploy): python scripts/check_environment.py --mode deploy
"""

import sys
import platform
import argparse


def check_python():
    """Check Python version."""
    version = sys.version_info
    print(f"  Python: {version.major}.{version.minor}.{version.micro}", end="")
    if version.major == 3 and version.minor >= 11:
        print(" ✅")
        return True
    else:
        print(" ❌ (Python 3.11+ gerekli)")
        return False


def check_pytorch():
    """Check PyTorch installation and device availability."""
    try:
        import torch
        print(f"  PyTorch: {torch.__version__} ✅")

        # MPS check (Apple Silicon)
        if torch.backends.mps.is_available():
            print("  Device: MPS (Apple Silicon) ✅")
            device = "mps"
        elif torch.cuda.is_available():
            print(f"  Device: CUDA ({torch.cuda.get_device_name(0)}) ✅")
            device = "cuda"
        else:
            print("  Device: CPU (no GPU acceleration) ⚠️")
            device = "cpu"

        # Quick tensor test
        test_tensor = torch.zeros(3, 224, 224).to(device)
        print(f"  Tensor test on {device}: ✅")
        return True, device
    except ImportError:
        print("  PyTorch: ❌ NOT INSTALLED")
        return False, "none"


def check_clip():
    """Check CLIP installation."""
    try:
        import clip
        print(f"  CLIP: ✅")
        return True
    except ImportError:
        print("  CLIP: ❌ NOT INSTALLED (run: pip install git+https://github.com/openai/CLIP.git)")
        return False


def check_transformers():
    """Check HuggingFace transformers."""
    try:
        import transformers
        print(f"  Transformers: {transformers.__version__} ✅")
        return True
    except ImportError:
        print("  Transformers: ❌ NOT INSTALLED")
        return False


def check_gradio():
    """Check Gradio for UI."""
    try:
        import gradio as gr
        print(f"  Gradio: {gr.__version__} ✅")
        return True
    except ImportError:
        print("  Gradio: ❌ NOT INSTALLED (deployment only)")
        return False


def check_training_deps():
    """Check training-specific dependencies."""
    deps = {
        "datasets": "HuggingFace datasets",
        "nltk": "BLEU score",
        "rouge_score": "ROUGE score",
        "sacrebleu": "Academic BLEU",
        "pycocotools": "COCO dataset tools",
        "tensorboard": "Training visualization",
        "tqdm": "Progress bars",
    }
    all_ok = True
    for module, desc in deps.items():
        try:
            __import__(module)
            print(f"  {desc} ({module}): ✅")
        except ImportError:
            print(f"  {desc} ({module}): ❌ NOT INSTALLED")
            all_ok = False
    return all_ok


def check_env_file():
    """Check .env configuration."""
    import os
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        print("  .env file: ✅")
        from dotenv import load_dotenv
        load_dotenv(env_path)
        hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        if hf_token and hf_token != "your-hf-token":
            print("  HuggingFace Token: ✅ (configured)")
        else:
            print("  HuggingFace Token: ⚠️ (not set - needed for upload)")
    else:
        print("  .env file: ⚠️ (.env.example'ı kopyalayıp .env olarak kaydet)")


def print_system_info():
    """Print system information."""
    print(f"\n{'='*50}")
    print("SYSTEM INFORMATION")
    print(f"{'='*50}")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Story Teller - Environment Checker")
    parser.add_argument("--mode", choices=["train", "deploy", "all"], default="all",
                        help="train: Mac training check | deploy: Windows UI check | all: everything")
    args = parser.parse_args()

    print_system_info()
    print("CHECKING ENVIRONMENT...")
    print(f"Mode: {args.mode.upper()}\n")

    results = {}

    print("[Core Dependencies]")
    results["python"] = check_python()
    results["pytorch"], device = check_pytorch()
    results["clip"] = check_clip()
    results["transformers"] = check_transformers()

    if args.mode in ("deploy", "all"):
        print("\n[Deployment Dependencies]")
        results["gradio"] = check_gradio()

    if args.mode in ("train", "all"):
        print("\n[Training Dependencies]")
        results["training"] = check_training_deps()

    print("\n[Configuration]")
    check_env_file()

    # Summary
    failed = [k for k, v in results.items() if not v]
    print(f"\n{'='*50}")
    if not failed:
        print("✅ Tüm bağımlılıklar hazır! Projeye başlayabilirsin.")
    else:
        print(f"❌ {len(failed)} sorun bulundu: {', '.join(failed)}")
        print("   requirements dosyasını tekrar yükle ve tekrar çalıştır.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
