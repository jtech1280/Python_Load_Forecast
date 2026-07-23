param(
    [string]$RunLabel = "",
    [switch]$DisableFiveMinLoad,
    [switch]$UseLocalWeatherCalibration,
    [string]$FixedOriginsFile = "",
    [int]$ReplayMaxOrigins = 0,
    [string]$PythonExe = "",
    [switch]$UpdateEnvironment,
    [switch]$SkipBootstrap
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
& (Join-Path $PSScriptRoot "import_forecast_env.ps1") -RepoRoot $RepoRoot

if ([string]::IsNullOrWhiteSpace($RunLabel)) {
    $RunLabel = "overnight_" + (Get-Date -Format "yyyyMMdd_HHmmss")
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

$OutputDir = Join-Path $RepoRoot "forecast_outputs"
if (![string]::IsNullOrWhiteSpace($env:FORECAST_OUTPUT_DIR)) {
    $OutputDir = $env:FORECAST_OUTPUT_DIR
    if (![System.IO.Path]::IsPathRooted($OutputDir)) {
        $OutputDir = Join-Path $RepoRoot $OutputDir
    }
}
$LogDir = Join-Path $OutputDir "run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogPath = Join-Path $LogDir "full_replay_$RunLabel.log"
$StatusPath = Join-Path $LogDir "full_replay_$RunLabel.status.json"
$FiveMinArg = $(if ($DisableFiveMinLoad) { " --disable-five-min-load" } else { "" })
$LocalWeatherArg = $(if ($UseLocalWeatherCalibration) { " --use-local-weather-calibration" } else { "" })
$FixedOriginsArg = $(if ([string]::IsNullOrWhiteSpace($FixedOriginsFile)) { "" } else { " --replay-fixed-origins-file `"$FixedOriginsFile`"" })
$ReplayMaxOriginsArg = $(if ($ReplayMaxOrigins -gt 0) { " --replay-max-origins $ReplayMaxOrigins" } else { "" })
$MainCommand = "`"$PythonExe`" -u -m forecasting.main --save-csv --rolling-origin-replay --safe-performance$FiveMinArg$LocalWeatherArg$FixedOriginsArg$ReplayMaxOriginsArg"

$startedAt = Get-Date
@{
    run_label = $RunLabel
    status = "running"
    started_at = $startedAt.ToString("o")
    command = $MainCommand
    validation_command = "`"$PythonExe`" -u scripts\validate_weather_interval_coverage.py --replay-path forecast_outputs\rolling_origin_replay_results.csv --output-label $RunLabel"
    log_path = $LogPath
    five_min_load_enabled = !$DisableFiveMinLoad
    local_weather_calibration_enabled = [bool]$UseLocalWeatherCalibration
    fixed_origins_file = $FixedOriginsFile
    replay_max_origins = $ReplayMaxOrigins
} | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

try {
    $pythonCmd = "$MainCommand > `"$LogPath`" 2>&1"
    & cmd.exe /d /c $pythonCmd
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        $artifacts = @(
            "rolling_origin_replay_results.csv",
            "rolling_origin_replay_origin_coverage.csv",
            "rolling_origin_replay_scorecard.csv",
            "rolling_origin_replay_summary.json",
            "rolling_origin_replay_stage_metrics.csv",
            "rolling_origin_replay_hot_peak_metrics_by_stage.csv",
            "rolling_origin_replay_shoulder_heat_metrics_by_stage.csv",
            "rolling_origin_replay_cloud_solar_midday_metrics_by_stage.csv",
            "rolling_origin_replay_long_horizon_metrics_by_stage.csv",
            "rolling_origin_replay_weather_realism_scorecard.csv",
            "rolling_origin_replay_weather_input_error_by_lead.csv",
            "rolling_origin_replay_weather_input_sensitivity_scorecard.csv",
            "rolling_origin_replay_weather_input_sensitivity_detail.csv",
            "production_readiness_scorecard.csv",
            "forecast_stage_metrics.csv",
            "feature_importance.csv",
            "model_features.txt",
            "local_weather_temperature_bias_summary.csv",
            "local_weather_temperature_bias_lookup.csv",
            "backtest_metrics_final.json",
            "band_coverage_summary.csv"
        )

        foreach ($name in $artifacts) {
            $src = Join-Path $OutputDir $name
            if (Test-Path $src) {
                $destName = [System.IO.Path]::GetFileNameWithoutExtension($name) + "_" + $RunLabel + [System.IO.Path]::GetExtension($name)
                Copy-Item -Path $src -Destination (Join-Path $OutputDir $destName) -Force
            }
        }

        "`nRunning weather scenario/conformal interval validation for $RunLabel..." | Add-Content -Path $LogPath -Encoding UTF8
        & $PythonExe -u scripts\validate_weather_interval_coverage.py `
            --replay-path (Join-Path $OutputDir "rolling_origin_replay_results.csv") `
            --output-label $RunLabel >> $LogPath 2>&1
        $validationExitCode = $LASTEXITCODE
        if ($validationExitCode -ne 0) {
            $exitCode = $validationExitCode
        }
    }

    $finishedAt = Get-Date
    @{
        run_label = $RunLabel
        status = $(if ($exitCode -eq 0) { "completed" } else { "failed" })
        exit_code = $exitCode
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        elapsed_minutes = [Math]::Round(($finishedAt - $startedAt).TotalMinutes, 2)
        log_path = $LogPath
        five_min_load_enabled = !$DisableFiveMinLoad
        local_weather_calibration_enabled = [bool]$UseLocalWeatherCalibration
        fixed_origins_file = $FixedOriginsFile
        replay_max_origins = $ReplayMaxOrigins
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

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
        log_path = $LogPath
        five_min_load_enabled = !$DisableFiveMinLoad
        local_weather_calibration_enabled = [bool]$UseLocalWeatherCalibration
        fixed_origins_file = $FixedOriginsFile
        replay_max_origins = $ReplayMaxOrigins
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
    throw
}
