param(
    [string]$RunLabel = "",
    [string]$FixedOriginsFile = "forecast_outputs\fixed_replay_origins_hot_peak_summers_20260531.txt",
    [int]$ReplayMaxOrigins = 20,
    [ValidateSet("safe", "gpu-priority", "cpu-only")]
    [string]$BackendMode = "safe",
    [string]$PythonExe = "",
    [switch]$UpdateEnvironment,
    [switch]$SkipBootstrap,
    [switch]$DisableProphet,
    [switch]$DisableCatBoost,
    [switch]$SkipDiagnostics
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
& (Join-Path $PSScriptRoot "import_forecast_env.ps1") -RepoRoot $RepoRoot

if ([string]::IsNullOrWhiteSpace($RunLabel)) {
    $RunLabel = "server_replay_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$usingDefaultPython = [string]::IsNullOrWhiteSpace($PythonExe)
if ($usingDefaultPython) {
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (($usingDefaultPython -and !$SkipBootstrap) -and ((!(Test-Path $PythonExe)) -or $UpdateEnvironment)) {
    & (Join-Path $PSScriptRoot "setup_forecast_environment.ps1")
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
    disable_prophet = [bool]$DisableProphet
    disable_catboost = [bool]$DisableCatBoost
    skip_diagnostics = [bool]$SkipDiagnostics
    fixed_origins_file = $FixedOriginsFile
    replay_max_origins = $ReplayMaxOrigins
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
        disable_prophet = [bool]$DisableProphet
        disable_catboost = [bool]$DisableCatBoost
        skip_diagnostics = [bool]$SkipDiagnostics
        fixed_origins_file = $FixedOriginsFile
        replay_max_origins = $ReplayMaxOrigins
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
        fixed_origins_file = $FixedOriginsFile
        replay_max_origins = $ReplayMaxOrigins
        log_path = $LogPath
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
    throw
}
