param(
    [string]$RunLabel = "",
    [string]$FixedOriginsFile = "",
    [int]$ReplayMaxOrigins = 20,
    [ValidateSet("safe", "gpu-priority", "cpu-only")]
    [string]$BackendMode = "safe",
    [ValidateSet("server", "gpu-cu12")]
    [string]$SetupProfile = "server",
    [string]$ServerConfigLocal = "",
    [string]$CudaDevice = "0",
    [int]$ReplayProcesses = 0,
    [string]$PythonExe = "",
    [switch]$UpdateEnvironment,
    [switch]$ForceRecreateVenv,
    [switch]$SkipBootstrap,
    [switch]$DisableProphet,
    [switch]$DisableCatBoost,
    [switch]$NoSaveSql,
    [switch]$SkipDiagnostics
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
elseif (![string]::IsNullOrWhiteSpace($env:FORECAST_CONFIG_LOCAL) -and
    [System.IO.Path]::GetFileName($env:FORECAST_CONFIG_LOCAL).Equals("config.server_rolling_origin_ada.yaml", [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Warning "Ignoring stale Ada server FORECAST_CONFIG_LOCAL for generic replay. Pass -ServerConfigLocal to use it explicitly."
    Remove-Item Env:\FORECAST_CONFIG_LOCAL -ErrorAction SilentlyContinue
}

if (![string]::IsNullOrWhiteSpace($CudaDevice)) {
    $env:FORECAST_CUDA_DEVICE = $CudaDevice
    $env:CUDA_VISIBLE_DEVICES = $CudaDevice
}

if ([string]::IsNullOrWhiteSpace($RunLabel)) {
    $RunLabel = "server_replay_" + (Get-Date -Format "yyyyMMdd_HHmmss")
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
    throw "Python executable exists but cannot run: $PythonExe. If this repo was copied from another machine, delete/recreate .venv on this server or pass -PythonExe to a working interpreter. Probe output: $probeText"
}

if (![string]::IsNullOrWhiteSpace($FixedOriginsFile) -and !(Test-Path $FixedOriginsFile)) {
    throw "Fixed origins file not found: $FixedOriginsFile"
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

$LogPath = Join-Path $LogDir "server_replay_$RunLabel.log"
$StatusPath = Join-Path $LogDir "server_replay_$RunLabel.status.json"

$argsList = @(
    "-u",
    "-m", "forecasting.main",
    "--save-csv",
    "--rolling-origin-replay"
)

if ($ReplayMaxOrigins -gt 0) {
    $argsList += @("--replay-max-origins", "$ReplayMaxOrigins")
}
if ($ReplayProcesses -gt 0) {
    $argsList += @("--replay-processes", "$ReplayProcesses")
}
if (![string]::IsNullOrWhiteSpace($FixedOriginsFile)) {
    $argsList += @("--replay-fixed-origins-file", $FixedOriginsFile)
}

switch ($BackendMode) {
    "safe" { $argsList += "--safe-performance" }
    "gpu-priority" { $argsList += "--gpu-priority" }
    "cpu-only" { $argsList += "--cpu-only" }
}

if ($DisableProphet) { $argsList += "--disable-prophet" }
if ($DisableCatBoost) { $argsList += "--disable-catboost" }
if ($NoSaveSql) { $argsList += "--no-save-sql" }
if ($SkipDiagnostics) { $argsList += "--skip-diagnostics" }

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
    disable_prophet = [bool]$DisableProphet
    disable_catboost = [bool]$DisableCatBoost
    no_save_sql = [bool]$NoSaveSql
    skip_diagnostics = [bool]$SkipDiagnostics
    fixed_origins_file = $FixedOriginsFile
    replay_max_origins = $ReplayMaxOrigins
    replay_processes = $ReplayProcesses
    log_path = $LogPath
} | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

Write-Host "Starting replay: $RunLabel"
Write-Host "Log: $LogPath"
Write-Host "Status: $StatusPath"

try {
    $cmdLine = "$commandText > `"$LogPath`" 2>&1"
    & cmd.exe /d /c $cmdLine
    $exitCode = $LASTEXITCODE

    $artifacts = @(
        "production_readiness_scorecard.csv",
        "rolling_origin_replay_results.csv",
        "rolling_origin_replay_scorecard.csv",
        "rolling_origin_replay_stage_metrics.csv",
        "rolling_origin_replay_hot_peak_metrics_by_stage.csv",
        "rolling_origin_replay_hot_peak_candidate_metrics_by_stage.csv",
        "rolling_origin_replay_hot_peak_candidate_scorecard.csv",
        "rolling_origin_replay_hot_ramp_peak_metrics_by_stage.csv",
        "rolling_origin_replay_hot_ramp_peak_candidate_scorecard.csv",
        "rolling_origin_replay_heat_persistence_peak_metrics_by_stage.csv",
        "rolling_origin_replay_heat_persistence_peak_candidate_scorecard.csv",
        "rolling_origin_replay_peak_window_metrics_by_stage.csv",
        "rolling_origin_replay_cloud_solar_midday_metrics_by_stage.csv",
        "backtest_metrics.json",
        "backtest_metrics_final.json",
        "model_features.txt",
        "prophet_regressor_features.txt",
        "xgb_training_backend.json",
        "lgb_training_backend.json",
        "catboost_training_backend.json",
        "runtime_performance.json",
        "diagnostics_manifest.json"
    )

    if ($exitCode -eq 0) {
        foreach ($name in $artifacts) {
            $src = Join-Path $OutputDir $name
            if (Test-Path $src) {
                $stem = [System.IO.Path]::GetFileNameWithoutExtension($name)
                $ext = [System.IO.Path]::GetExtension($name)
                Copy-Item -Path $src -Destination (Join-Path $OutputDir "$stem`_$RunLabel$ext") -Force
            }
        }

        & $PythonExe scripts\summarize_replay_scorecard.py --label $RunLabel *> (Join-Path $LogDir "server_replay_$RunLabel.summary.txt")
    }

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
        disable_prophet = [bool]$DisableProphet
        disable_catboost = [bool]$DisableCatBoost
        no_save_sql = [bool]$NoSaveSql
        skip_diagnostics = [bool]$SkipDiagnostics
        fixed_origins_file = $FixedOriginsFile
        replay_max_origins = $ReplayMaxOrigins
        replay_processes = $ReplayProcesses
        log_path = $LogPath
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

    Write-Host "Replay finished with exit code $exitCode"
    Write-Host "Summary: $(Join-Path $LogDir "server_replay_$RunLabel.summary.txt")"
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
        fixed_origins_file = $FixedOriginsFile
        replay_max_origins = $ReplayMaxOrigins
        replay_processes = $ReplayProcesses
        log_path = $LogPath
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
    throw
}
