param(
    [string]$RunLabel = "",
    [ValidateSet("safe", "gpu-priority", "cpu-only")]
    [string]$BackendMode = "safe",
    [ValidateSet("server", "gpu-cu12")]
    [string]$SetupProfile = "server",
    [string]$ServerConfigLocal = "",
    [string]$CudaDevice = "0",
    [string]$PythonExe = "",
    [switch]$UpdateEnvironment,
    [switch]$ForceRecreateVenv,
    [switch]$SkipBootstrap,
    [switch]$SkipSolarForecast,
    [switch]$AllowStaleSolarForecast,
    [switch]$NoSaveSql,
    [switch]$RollingOriginReplay,
    [int]$ReplayMaxOrigins = 0,
    [int]$ReplayProcesses = 0,
    [switch]$SkipDiagnostics,
    [switch]$RunDashboard,
    [int]$DashboardPort = 8050
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
& (Join-Path $PSScriptRoot "import_forecast_env.ps1") -RepoRoot $RepoRoot

if (![string]::IsNullOrWhiteSpace($ServerConfigLocal)) {
    if (![System.IO.Path]::IsPathRooted($ServerConfigLocal)) {
        $ServerConfigLocal = Join-Path $RepoRoot $ServerConfigLocal
    }
    if (!(Test-Path $ServerConfigLocal)) {
        throw "Server override config not found: $ServerConfigLocal"
    }
    $env:FORECAST_CONFIG_LOCAL = $ServerConfigLocal
}

if (![string]::IsNullOrWhiteSpace($CudaDevice)) {
    $env:FORECAST_CUDA_DEVICE = $CudaDevice
    $env:CUDA_VISIBLE_DEVICES = $CudaDevice
}

if ([string]::IsNullOrWhiteSpace($RunLabel)) {
    $RunLabel = "forecast_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$usingDefaultPython = [string]::IsNullOrWhiteSpace($PythonExe)
if ($usingDefaultPython) {
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}

function Test-PythonExeRuns {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        return $false
    }
    try {
        & $Path -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

if (($usingDefaultPython -and !$SkipBootstrap) -and ($ForceRecreateVenv -or $UpdateEnvironment -or !(Test-PythonExeRuns -Path $PythonExe))) {
    $setupArgs = @{ Profile = $SetupProfile }
    if ($ForceRecreateVenv) {
        $setupArgs.ForceRecreate = $true
    }
    & (Join-Path $PSScriptRoot "setup_forecast_environment.ps1") @setupArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Environment bootstrap failed."
    }
}

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe. Run scripts\setup_forecast_environment.ps1 or pass -PythonExe C:\path\to\python.exe."
}

$pythonProbe = & $PythonExe -c "import sys; print(sys.executable)" 2>&1
if ($LASTEXITCODE -ne 0) {
    $probeText = ($pythonProbe | Out-String).Trim()
    throw "Python executable exists but cannot run: $PythonExe. Probe output: $probeText"
}

$OutputDir = Join-Path $RepoRoot "forecast_outputs"
if (![string]::IsNullOrWhiteSpace($env:FORECAST_OUTPUT_DIR)) {
    $OutputDir = $env:FORECAST_OUTPUT_DIR
    if (![System.IO.Path]::IsPathRooted($OutputDir)) {
        $OutputDir = Join-Path $RepoRoot $OutputDir
    }
}
$LogDir = Join-Path $OutputDir "run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogPath = Join-Path $LogDir "forecast_$RunLabel.log"
$StatusPath = Join-Path $LogDir "forecast_$RunLabel.status.json"

$argsList = @(
    "-u",
    "-m", "forecasting.main",
    "--save-csv"
)

switch ($BackendMode) {
    "safe" { $argsList += "--safe-performance" }
    "gpu-priority" { $argsList += "--gpu-priority" }
    "cpu-only" { $argsList += "--cpu-only" }
}

if ($SkipSolarForecast) { $argsList += "--skip-solar-forecast" }
if ($AllowStaleSolarForecast) { $argsList += "--allow-stale-solar-forecast" }
if ($NoSaveSql) { $argsList += "--no-save-sql" }
if ($RollingOriginReplay) { $argsList += "--rolling-origin-replay" }
if ($ReplayMaxOrigins -gt 0) { $argsList += @("--replay-max-origins", "$ReplayMaxOrigins") }
if ($ReplayProcesses -gt 0) { $argsList += @("--replay-processes", "$ReplayProcesses") }
if ($SkipDiagnostics) { $argsList += "--skip-diagnostics" }
if ($RunDashboard) { $argsList += @("--run-dashboard", "--dashboard-port", "$DashboardPort") }

$startedAt = Get-Date
$commandText = "`"$PythonExe`" " + (($argsList | ForEach-Object {
    if ($_ -match "\s") { "`"$_`"" } else { $_ }
}) -join " ")

@{
    run_label = $RunLabel
    status = "running"
    started_at = $startedAt.ToString("o")
    command = $commandText
    backend_mode = $BackendMode
    setup_profile = $SetupProfile
    force_recreate_venv = [bool]$ForceRecreateVenv
    config_local = $env:FORECAST_CONFIG_LOCAL
    cuda_device = $env:FORECAST_CUDA_DEVICE
    data_root = $env:FORECAST_DATA_ROOT
    solar_parquet_root = $env:FORECAST_SOLAR_PARQUET_ROOT
    log_path = $LogPath
    output_dir = $OutputDir
    no_save_sql = [bool]$NoSaveSql
    skip_solar_forecast = [bool]$SkipSolarForecast
    allow_stale_solar_forecast = [bool]$AllowStaleSolarForecast
    rolling_origin_replay = [bool]$RollingOriginReplay
    replay_max_origins = $ReplayMaxOrigins
    replay_processes = $ReplayProcesses
} | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

Write-Host "Starting forecast: $RunLabel"
Write-Host "Log: $LogPath"
Write-Host "Status: $StatusPath"

try {
    & $PythonExe @argsList *> $LogPath
    $exitCode = $LASTEXITCODE

    $finishedAt = Get-Date
    @{
        run_label = $RunLabel
        status = $(if ($exitCode -eq 0) { "completed" } else { "failed" })
        exit_code = $exitCode
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        elapsed_minutes = [Math]::Round(($finishedAt - $startedAt).TotalMinutes, 2)
        command = $commandText
        backend_mode = $BackendMode
        setup_profile = $SetupProfile
        force_recreate_venv = [bool]$ForceRecreateVenv
        config_local = $env:FORECAST_CONFIG_LOCAL
        cuda_device = $env:FORECAST_CUDA_DEVICE
        data_root = $env:FORECAST_DATA_ROOT
        solar_parquet_root = $env:FORECAST_SOLAR_PARQUET_ROOT
        log_path = $LogPath
        output_dir = $OutputDir
        no_save_sql = [bool]$NoSaveSql
        skip_solar_forecast = [bool]$SkipSolarForecast
        allow_stale_solar_forecast = [bool]$AllowStaleSolarForecast
        rolling_origin_replay = [bool]$RollingOriginReplay
        replay_max_origins = $ReplayMaxOrigins
        replay_processes = $ReplayProcesses
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

    Write-Host "Forecast finished with exit code $exitCode"
    exit $exitCode
}
catch {
    $finishedAt = Get-Date
    @{
        run_label = $RunLabel
        status = "error"
        error = $_.Exception.Message
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        elapsed_minutes = [Math]::Round(($finishedAt - $startedAt).TotalMinutes, 2)
        command = $commandText
        backend_mode = $BackendMode
        setup_profile = $SetupProfile
        force_recreate_venv = [bool]$ForceRecreateVenv
        config_local = $env:FORECAST_CONFIG_LOCAL
        cuda_device = $env:FORECAST_CUDA_DEVICE
        data_root = $env:FORECAST_DATA_ROOT
        solar_parquet_root = $env:FORECAST_SOLAR_PARQUET_ROOT
        log_path = $LogPath
        output_dir = $OutputDir
        rolling_origin_replay = [bool]$RollingOriginReplay
        replay_max_origins = $ReplayMaxOrigins
        replay_processes = $ReplayProcesses
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
    throw
}
