param(
    [string]$PythonExe = "",
    [ValidateSet("server", "gpu-cu12")]
    [string]$Profile = "server",
    [switch]$SkipInstall,
    [switch]$SkipGpuValidation
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RequirementsByProfile = @{
    "server" = (Join-Path $RepoRoot "requirements-server-lock.txt")
    "gpu-cu12" = (Join-Path $RepoRoot "requirements-gpu-cu12-lock.txt")
}
$Requirements = $RequirementsByProfile[$Profile]

if (!(Test-Path $Requirements)) {
    throw "Requirements lock file not found: $Requirements"
}

function New-PythonCandidate {
    param(
        [string]$File,
        [string[]]$Args = @()
    )
    [pscustomobject]@{
        File = $File
        Args = $Args
    }
}

function Test-PythonCandidate {
    param($Candidate)
    $probeArgs = @($Candidate.Args) + @(
        "-c",
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
    )
    & $Candidate.File @probeArgs *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-BasePython {
    $candidates = New-Object System.Collections.Generic.List[object]

    if (![string]::IsNullOrWhiteSpace($PythonExe)) {
        $candidates.Add((New-PythonCandidate -File $PythonExe))
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $candidates.Add((New-PythonCandidate -File $pyLauncher.Source -Args @("-3.12")))
        $candidates.Add((New-PythonCandidate -File $pyLauncher.Source -Args @("-3")))
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates.Add((New-PythonCandidate -File $pythonCommand.Source))
    }

    foreach ($candidate in $candidates) {
        if (Test-PythonCandidate $candidate) {
            return $candidate
        }
    }

    throw "No usable Python 3.12+ interpreter found. Install Python 3.12+ or pass -PythonExe C:\path\to\python.exe."
}

function Write-GpuDriverProbe {
    if ($Profile -ne "gpu-cu12") {
        return
    }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (!$nvidiaSmi) {
        Write-Warning "nvidia-smi was not found. Install/update the NVIDIA driver before expecting CuPy/XGBoost/CatBoost GPU validation to pass."
        return
    }

    $gpuInfo = & $nvidiaSmi.Source --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpuInfo) {
        Write-Host "NVIDIA GPU/driver detected:"
        $gpuInfo | ForEach-Object { Write-Host "  $_" }
    }
    else {
        Write-Warning "nvidia-smi is installed but did not report a usable NVIDIA GPU/driver."
    }
}

Write-GpuDriverProbe

if (!(Test-Path $VenvPython)) {
    $basePython = Resolve-BasePython
    Write-Host "Creating virtual environment: $VenvPython"
    $venvArgs = @($basePython.Args) + @("-m", "venv", ".venv")
    & $basePython.File @venvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
}

if (!(Test-Path $VenvPython)) {
    throw "Virtual environment Python was not created at $VenvPython"
}

if (!$SkipInstall) {
    Write-Host "Installing locked dependencies for profile '$Profile' from $Requirements"
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed."
    }

    & $VenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency install failed."
    }
}

$validationCode = @'
import importlib
import sys
import warnings

profile = sys.argv[1]
skip_gpu_validation = sys.argv[2].lower() in {'1', 'true', 'yes'}

if profile == 'gpu-cu12':
    warnings.filterwarnings(
        'ignore',
        message='CUDA path could not be detected.*',
        category=UserWarning,
        module=r'cupy\._environment',
    )

required_modules = [
    'catboost',
    'dash',
    'joblib',
    'lightgbm',
    'numpy',
    'pandas',
    'plotly',
    'prophet',
    'pyarrow',
    'pyodbc',
    'requests',
    'sklearn',
    'sqlalchemy',
    'xgboost',
    'yaml',
]

if profile == 'gpu-cu12':
    required_modules.extend(['cupy', 'optuna', 'pytest'])

missing = []
for module_name in required_modules:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f'{module_name}: {exc}')

if missing:
    raise SystemExit('Virtual environment validation failed:\n' + '\n'.join(missing))

if profile == 'gpu-cu12':
    import cupy as cp

    print(f'CuPy {cp.__version__} installed.')
    if not skip_gpu_validation:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise SystemExit('CuPy is installed, but no CUDA device is visible.')
        smoke = cp.arange(4, dtype=cp.float32).sum()
        if float(cp.asnumpy(smoke)) != 6.0:
            raise SystemExit('CuPy CUDA smoke test returned an unexpected result.')
        print(f'CuPy CUDA smoke test ok; visible CUDA devices={device_count}.')

print(f'forecast venv ok ({profile})')
'@

$skipGpuValidationText = ([bool]$SkipGpuValidation).ToString()
& $VenvPython -c $validationCode $Profile $skipGpuValidationText
if ($LASTEXITCODE -ne 0) {
    throw "Virtual environment validation failed."
}

Write-Host "Forecast Python: $VenvPython"
Write-Output $VenvPython
