param(
    [string]$PythonExe = "",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $RepoRoot "requirements-server-lock.txt"

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
    Write-Host "Installing locked dependencies from $Requirements"
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed."
    }

    & $VenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency install failed."
    }
}

& $VenvPython -c "import pandas, numpy, sklearn, xgboost, lightgbm, catboost, yaml, pyodbc, pyarrow; print('forecast venv ok')"
if ($LASTEXITCODE -ne 0) {
    throw "Virtual environment validation failed."
}

Write-Host "Forecast Python: $VenvPython"
Write-Output $VenvPython
