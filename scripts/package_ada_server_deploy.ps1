param(
    [string]$DestinationPath = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($DestinationPath)) {
    $packageDir = Join-Path $RepoRoot "forecast_outputs\deploy_packages"
    New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
    $DestinationPath = Join-Path $packageDir ("load_forecast_ada_server_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".zip")
}
elseif (![System.IO.Path]::IsPathRooted($DestinationPath)) {
    $DestinationPath = Join-Path $RepoRoot $DestinationPath
}

$destinationFull = [System.IO.Path]::GetFullPath($DestinationPath)
$destinationDir = Split-Path -Parent $destinationFull
New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

$excludedDirs = @(
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "forecast_outputs",
    "weather_cache"
)
$excludedFilePatterns = @("*.pyc", "*.pyo")
$excludedFileNames = @(".env", ".env.local")
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("forecast_ada_zip_" + [System.Guid]::NewGuid().ToString("N"))
$repoFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd("\")

function Get-RepoRelativePath {
    param([string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    $prefix = $repoFull + [System.IO.Path]::DirectorySeparatorChar
    if (!$full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is not inside repository root: $full"
    }
    return $full.Substring($prefix.Length)
}

try {
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

    $files = Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Force | Where-Object {
        $relative = Get-RepoRelativePath -Path $_.FullName
        $parts = $relative -split "[\\/]+"
        $excluded = $false
        foreach ($part in $parts) {
            if ($excludedDirs -contains $part) {
                $excluded = $true
                break
            }
        }
        if (!$excluded) {
            if ($excludedFileNames -contains $_.Name) {
                $excluded = $true
            }
        }
        if (!$excluded) {
            foreach ($pattern in $excludedFilePatterns) {
                if ($_.Name -like $pattern) {
                    $excluded = $true
                    break
                }
            }
        }
        !$excluded
    }

    foreach ($file in $files) {
        $relative = Get-RepoRelativePath -Path $file.FullName
        $dest = Join-Path $tempRoot $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
    }

    if (Test-Path $destinationFull) {
        Remove-Item -LiteralPath $destinationFull -Force
    }
    Compress-Archive -Path (Join-Path $tempRoot "*") -DestinationPath $destinationFull -Force
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host "Created Ada server deployment package:"
Write-Host "  $destinationFull"
