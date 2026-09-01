param(
    [string]$RepoRoot = ""
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

foreach ($name in @(".env", ".env.local")) {
    $path = Join-Path $RepoRoot $name
    if (!(Test-Path $path)) {
        continue
    }

    foreach ($rawLine in Get-Content -Path $path) {
        $line = $rawLine.Trim()
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
            continue
        }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }

        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            continue
        }

        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim("'`"")
        if ([string]::IsNullOrWhiteSpace($key)) {
            continue
        }
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($key, "Process"))) {
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

function Set-DefaultIfEmpty {
    param(
        [string]$Name,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }
    if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($Name, "Process"))) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable("FORECAST_DATA_ROOT", "Process")) -and
    [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable("FORECAST_SOLAR_PARQUET_ROOT", "Process"))) {
    foreach ($candidate in @("D:\PY_LRS", "C:\PY_LRS")) {
        if (Test-Path $candidate) {
            Set-DefaultIfEmpty -Name "FORECAST_DATA_ROOT" -Value $candidate
            Set-DefaultIfEmpty -Name "FORECAST_SOLAR_PARQUET_ROOT" -Value $candidate
            break
        }
    }
}
elseif ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable("FORECAST_SOLAR_PARQUET_ROOT", "Process"))) {
    Set-DefaultIfEmpty `
        -Name "FORECAST_SOLAR_PARQUET_ROOT" `
        -Value ([Environment]::GetEnvironmentVariable("FORECAST_DATA_ROOT", "Process"))
}
