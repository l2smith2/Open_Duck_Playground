param(
    [ValidateSet("SmokeBenchmark")]
    [string]$Mode = "SmokeBenchmark",
    [ValidateSet("NvidiaTeslaT4", "NvidiaTeslaP100")]
    [string]$Accelerator = "NvidiaTeslaT4",
    [string]$Slug = "open-duck-free-first-bdx-benchmark",
    [string]$KaggleUser = "",
    [switch]$NoPush,
    [switch]$Poll,
    [switch]$Download
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$branch = "codex/free-first-bdx-policy"
$batchRoot = Join-Path $repoRoot ".kaggle-api"
$batchDir = Join-Path $batchRoot $Slug
$outputRoot = Join-Path $repoRoot "kaggle-outputs"

function Invoke-Kaggle {
    param(
        [Parameter(Mandatory=$true)]
        [string[]]$Arguments,
        [switch]$Capture
    )
    if ($script:KaggleExe) {
        if ($Capture) { $result = & $script:KaggleExe @Arguments } else { & $script:KaggleExe @Arguments }
    } else {
        if ($Capture) { $result = & py -m kaggle @Arguments } else { & py -m kaggle @Arguments }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "kaggle $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
    if ($Capture) { return $result }
}

function ConvertTo-GitHubHttpsUrl {
    param([string]$RemoteUrl)
    if ($RemoteUrl -match '^https://github\.com/.+/.+(?:\.git)?$') {
        return $RemoteUrl
    }
    if ($RemoteUrl -match '^git@github\.com:(.+/.+(?:\.git)?)$') {
        return "https://github.com/$($Matches[1])"
    }
    throw "Git origin must be a GitHub fork URL. Found: $RemoteUrl"
}

function Get-KaggleUserFromCli {
    $csv = Invoke-Kaggle -Arguments @("kernels", "list", "-m", "--page-size", "1", "-v") -Capture
    $rows = @($csv | ConvertFrom-Csv)
    if ($rows.Count -gt 0 -and $rows[0].author) {
        return $rows[0].author
    }
    return ""
}

function New-BatchScript {
    param(
        [string]$ForkUrl,
        [string]$ExpectedUpstream,
        [string]$OutputPath
    )

    $template = @'
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

WORK = Path("/kaggle/working")
REPO = WORK / "Open_Duck_Playground"
ARTIFACTS = WORK / "artifacts"
FORK_URL = __FORK_URL_JSON__
POLICY_BRANCH = "codex/free-first-bdx-policy"
EXPECTED_UPSTREAM = "__EXPECTED_UPSTREAM__"
EXPECTED_STACK = {
    "brax": "0.13.0",
    "jax": "0.6.2",
    "jax-cuda12-plugin": "0.6.2",
    "jaxlib": "0.6.2",
    "mujoco": "3.3.3",
    "mujoco-mjx": "3.3.3",
    "playground": "0.0.5",
}


def run(command, cwd=None):
    print("RUN:", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def check_output(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True)


def configure_gpu_env():
    gpu_check = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True)
    if gpu_check.returncode != 0 or "GPU" not in gpu_check.stdout:
        raise RuntimeError("No NVIDIA GPU is attached to this Kaggle run")
    print(gpu_check.stdout.strip())
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    os.environ["PATH"] = "/usr/local/nvidia/bin:" + os.environ["PATH"]
    os.environ["UV_LINK_MODE"] = "copy"


def prepare_workspace():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "git-lfs"])
    run([sys.executable, "-m", "pip", "install", "uv"])

    if not REPO.exists():
        run(["git", "clone", "--branch", POLICY_BRANCH, FORK_URL, str(REPO)])
    else:
        run(["git", "fetch", "origin", POLICY_BRANCH], cwd=REPO)
        run(["git", "checkout", "-B", POLICY_BRANCH, "FETCH_HEAD"], cwd=REPO)

    run(["git", "lfs", "install"], cwd=REPO)
    run(["git", "lfs", "pull"], cwd=REPO)
    recorded = (REPO / "UPSTREAM_COMMIT").read_text().strip()
    if recorded != EXPECTED_UPSTREAM:
        raise AssertionError((recorded, EXPECTED_UPSTREAM))

    venv_dir = REPO / ".venv"
    venv_python = venv_dir / "bin" / "python"
    if venv_dir.exists() and (not venv_python.is_file() or not os.access(venv_python, os.X_OK)):
        print("Repairing persisted virtual environment...")
        run(["uv", "venv", "--clear", "--python", sys.executable, str(venv_dir)], cwd=REPO)
    run(["uv", "sync", "--locked"], cwd=REPO)


def verify_stack():
    backend_lines = check_output([
        "uv", "run", "python", "-c",
        "import jax; print(jax.default_backend()); print(jax.local_device_count())",
    ], cwd=REPO).splitlines()
    backend = backend_lines[0].strip()
    device_count = int(backend_lines[1])
    if backend != "gpu":
        raise AssertionError(f"Expected JAX GPU backend, got {backend!r}")
    if device_count < 1:
        raise AssertionError(f"Expected at least one JAX device, got {device_count}")

    stack_output = check_output([
        "uv", "run", "python", "-c",
        "import json; from importlib.metadata import version; "
        "from mujoco_playground._src.collision import geoms_colliding; "
        "from mujoco_playground import wrapper; "
        "from mujoco_playground.config import locomotion_params; "
        "packages = ('brax', 'jax', 'jaxlib', 'jax-cuda12-plugin', 'mujoco', 'mujoco-mjx', 'playground'); "
        "print('TRAINING_STACK=' + json.dumps({name: version(name) for name in packages}, sort_keys=True))",
    ], cwd=REPO)
    prefix = "TRAINING_STACK="
    lines = [line.removeprefix(prefix) for line in stack_output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise AssertionError(f"Could not identify training stack in output: {stack_output!r}")
    training_stack = json.loads(lines[0])
    if training_stack != EXPECTED_STACK:
        raise AssertionError(f"Expected {EXPECTED_STACK}, got {training_stack}")

    (ARTIFACTS / "workspace.json").write_text(json.dumps({
        "fork": FORK_URL,
        "branch": POLICY_BRANCH,
        "upstream": EXPECTED_UPSTREAM,
        "backend": backend,
        "device_count": device_count,
        "training_stack": training_stack,
    }, indent=2) + "\n")
    print("Ready:", backend, "devices:", device_count, ARTIFACTS)


def run_stage(name, steps, stage, seed, restore=None, imitation_scale=1.0):
    output = ARTIFACTS / name
    command = [
        "uv", "run", "python", "scripts/run_training_stage.py",
        "--name", name,
        "--output-dir", str(output),
        "--steps", str(steps),
        "--randomization-stage", stage,
        "--seed", str(seed),
        "--imitation-reward-weight-scale", str(imitation_scale),
    ]
    if restore:
        command += ["--restore", str(restore)]
    run(command, cwd=REPO)
    return json.loads((output / "stage_result.json").read_text())


def save_bundle(label):
    archive = shutil.make_archive(str(WORK / label), "zip", ARTIFACTS)
    print("Kaggle output bundle:", archive)
    return archive


configure_gpu_env()
prepare_workspace()
verify_stack()

smoke = run_stage("00_smoke_1m", 1_000_000, "nominal", 100)
benchmark = run_stage("01_neutral_nominal_20m", 20_000_000, "nominal", 101)
run([
    "uv", "run", "python", "scripts/estimate_compute.py",
    "--benchmark-seconds", str(benchmark["elapsed_seconds"]),
    "--output", str(ARTIFACTS / "compute_decision.json"),
], cwd=REPO)
decision = json.loads((ARTIFACTS / "compute_decision.json").read_text())
print(json.dumps(decision, indent=2))
save_bundle("after_benchmark")
'@

    $forkJson = $ForkUrl | ConvertTo-Json -Compress
    $template = $template.Replace("__FORK_URL_JSON__", $forkJson)
    $template = $template.Replace("__EXPECTED_UPSTREAM__", $ExpectedUpstream)
    [System.IO.File]::WriteAllText($OutputPath, $template, [System.Text.UTF8Encoding]::new($false))
}

Set-Location $repoRoot
Write-Host "Open Duck Kaggle API batch" -ForegroundColor Cyan
Write-Host "Repository: $repoRoot"

if ((git branch --show-current).Trim() -ne $branch) {
    throw "Expected branch $branch."
}

$script:KaggleExe = (Get-Command kaggle -ErrorAction SilentlyContinue).Source
if (-not $script:KaggleExe) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Kaggle CLI was not found. Install it with: py -m pip install --user kaggle"
    }
    & py -m kaggle --version
    if ($LASTEXITCODE -ne 0) { throw "Kaggle CLI is not available through py -m kaggle." }
} else {
    & $script:KaggleExe --version
    if ($LASTEXITCODE -ne 0) { throw "Kaggle CLI is installed but not usable." }
}

if (-not $KaggleUser) {
    $KaggleUser = Get-KaggleUserFromCli
}
if (-not $KaggleUser) {
    $KaggleUser = Read-Host "Kaggle username"
}
if (-not $KaggleUser) {
    throw "Kaggle username is required."
}

$origin = ConvertTo-GitHubHttpsUrl ((git remote get-url origin).Trim())
$expectedUpstream = (Get-Content -LiteralPath (Join-Path $repoRoot "UPSTREAM_COMMIT") -Raw).Trim()

Write-Host "Publishing $branch to the GitHub fork..."
git push --set-upstream origin $branch
if ($LASTEXITCODE -ne 0) { throw "Git push failed." }

New-Item -ItemType Directory -Path $batchDir -Force | Out-Null
$codeFile = Join-Path $batchDir "open_duck_free_first_bdx_batch.py"
New-BatchScript -ForkUrl $origin -ExpectedUpstream $expectedUpstream -OutputPath $codeFile

$metadata = [ordered]@{
    id = "$KaggleUser/$Slug"
    title = "Open Duck free-first BDX smoke benchmark"
    code_file = "open_duck_free_first_bdx_batch.py"
    language = "python"
    kernel_type = "script"
    is_private = $true
    enable_gpu = $true
    enable_internet = $true
    machine_shape = $Accelerator
    dataset_sources = @()
    competition_sources = @()
    kernel_sources = @()
    model_sources = @()
} | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText((Join-Path $batchDir "kernel-metadata.json"), $metadata + "`n", [System.Text.UTF8Encoding]::new($false))

Write-Host "Prepared ignored Kaggle batch folder: $batchDir" -ForegroundColor Green
Write-Host "Kernel: $KaggleUser/$Slug"
Write-Host "Accelerator: $Accelerator"

if (-not $NoPush) {
    Invoke-Kaggle -Arguments @("kernels", "push", "-p", $batchDir, "--accelerator", $Accelerator, "--timeout", "43200")
}

if ($Poll) {
    do {
        Start-Sleep -Seconds 60
        $status = Invoke-Kaggle -Arguments @("kernels", "status", "$KaggleUser/$Slug") -Capture
        $status | ForEach-Object { Write-Host $_ }
    } while (($status -join " ") -match "running|queued|pending")
}

if ($Download) {
    $downloadDir = Join-Path $outputRoot $Slug
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    Invoke-Kaggle -Arguments @("kernels", "output", "$KaggleUser/$Slug", "-p", $downloadDir, "-o")
    Write-Host "Downloaded Kaggle outputs to: $downloadDir" -ForegroundColor Green
}
