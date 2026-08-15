param(
    [string]$PythonExe = "",
    [switch]$ForceRecreate,
    [switch]$SkipInstall,
    [switch]$SkipGpuValidation
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot ".venv"

if ($ForceRecreate -and (Test-Path $VenvDir)) {
    $resolvedRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
    $resolvedVenv = (Resolve-Path -LiteralPath $VenvDir).Path
    $repoPrefix = $resolvedRepo.TrimEnd("\") + "\"

    if (!$resolvedVenv.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove venv outside repo: $resolvedVenv"
    }

    Write-Host "Removing existing virtual environment: $resolvedVenv"
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

$setupArgs = @{
    Profile = "gpu-cu12"
}
if (![string]::IsNullOrWhiteSpace($PythonExe)) {
    $setupArgs.PythonExe = $PythonExe
}
if ($SkipInstall) {
    $setupArgs.SkipInstall = $true
}
if ($SkipGpuValidation) {
    $setupArgs.SkipGpuValidation = $true
}

& (Join-Path $PSScriptRoot "setup_forecast_environment.ps1") @setupArgs
if (!$?) {
    throw "CUDA 12 environment rebuild failed."
}

Write-Host ""
Write-Host "Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Run the active performance analysis with:"
Write-Host "  python ActivePerformnaceAnalysis.py"
