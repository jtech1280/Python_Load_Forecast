param(
    [string]$RunLabel = "",
    [string]$ServerConfigLocal = "",
    [string]$OutputDir = "",
    [string]$CudaDevice = "0",
    [string]$PythonExe = "",
    [int]$ReplayMaxOrigins = 0,
    [int]$ReplayProcesses = 3,
    [string]$FixedOriginsFile = "",
    [switch]$UpdateEnvironment,
    [switch]$ForceRecreateVenv,
    [switch]$SkipBootstrap,
    [switch]$SkipSolarForecast,
    [switch]$AllowStaleSolarForecast,
    [switch]$SaveSql,
    [switch]$NoSaveSql,
    [switch]$SkipDiagnostics,
    [switch]$UseLightGbmGpu,
    [switch]$RunValidation,
    [switch]$SkipGpuPreflight,
    [int]$GpuMonitorIntervalSec = 30
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
& (Join-Path $PSScriptRoot "import_forecast_env.ps1") -RepoRoot $RepoRoot

if (![string]::IsNullOrWhiteSpace($CudaDevice)) {
    $env:FORECAST_CUDA_DEVICE = $CudaDevice
    $env:CUDA_VISIBLE_DEVICES = $CudaDevice
}
elseif (![string]::IsNullOrWhiteSpace($env:FORECAST_CUDA_DEVICE)) {
    $env:CUDA_VISIBLE_DEVICES = $env:FORECAST_CUDA_DEVICE
}

if ([string]::IsNullOrWhiteSpace($RunLabel)) {
    $RunLabel = "ada_rolling_origin_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

if ([string]::IsNullOrWhiteSpace($ServerConfigLocal)) {
    $ServerConfigLocal = Join-Path $RepoRoot "forecasting\config.server_rolling_origin_ada.yaml"
}
if (![System.IO.Path]::IsPathRooted($ServerConfigLocal)) {
    $ServerConfigLocal = Join-Path $RepoRoot $ServerConfigLocal
}
if (!(Test-Path $ServerConfigLocal)) {
    throw "Server override config not found: $ServerConfigLocal"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    if ([string]::IsNullOrWhiteSpace($env:FORECAST_OUTPUT_DIR)) {
        $OutputDir = Join-Path $RepoRoot "forecast_outputs\server_rolling_origin"
    }
    else {
        $OutputDir = $env:FORECAST_OUTPUT_DIR
    }
}
if (![System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

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
    $setupArgs = @{ Profile = "gpu-cu12" }
    if ($ForceRecreateVenv) {
        $setupArgs.ForceRecreate = $true
    }
    & (Join-Path $PSScriptRoot "setup_forecast_environment.ps1") @setupArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA environment bootstrap failed."
    }
}
if (!(Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe. Run scripts\setup_forecast_environment.ps1 -Profile gpu-cu12 or pass -PythonExe C:\path\to\python.exe."
}

$pythonProbe = & $PythonExe -c "import sys; print(sys.executable)" 2>&1
if ($LASTEXITCODE -ne 0) {
    $probeText = ($pythonProbe | Out-String).Trim()
    throw "Python executable exists but cannot run: $PythonExe. Probe output: $probeText"
}

if (![string]::IsNullOrWhiteSpace($FixedOriginsFile)) {
    if (![System.IO.Path]::IsPathRooted($FixedOriginsFile)) {
        $FixedOriginsFile = Join-Path $RepoRoot $FixedOriginsFile
    }
    if (!(Test-Path $FixedOriginsFile)) {
        throw "Fixed origins file not found: $FixedOriginsFile"
    }
}

if ($SaveSql -and $NoSaveSql) {
    throw "Pass only one of -SaveSql or -NoSaveSql."
}

$env:FORECAST_CONFIG_LOCAL = $ServerConfigLocal
$env:FORECAST_OUTPUT_DIR = $OutputDir
$env:FORECAST_CATBOOST_REQUIRE_GPU = "true"
$env:FORECAST_CATBOOST_TASK_TYPE = "GPU"
if ([string]::IsNullOrWhiteSpace($env:FORECAST_CUDA_DEVICE)) {
    $env:FORECAST_CUDA_DEVICE = "0"
    $env:CUDA_VISIBLE_DEVICES = "0"
}
$env:PYTHONUNBUFFERED = "1"
$env:FORECAST_ENABLE_PLATFORM_WMI = "0"

$LogDir = Join-Path $OutputDir "run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogPath = Join-Path $LogDir "rolling_origin_replay_$RunLabel.log"
$StatusPath = Join-Path $LogDir "rolling_origin_replay_$RunLabel.status.json"
$SummaryPath = Join-Path $LogDir "rolling_origin_replay_$RunLabel.summary.txt"
$GpuMonitorPath = Join-Path $LogDir "rolling_origin_replay_$RunLabel.gpu.csv"

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
if ($SkipSolarForecast) { $argsList += "--skip-solar-forecast" }
if ($AllowStaleSolarForecast) { $argsList += "--allow-stale-solar-forecast" }
if ($SaveSql) {
    $argsList += "--save-sql"
}
else {
    # Server replay is a performance/diagnostic workflow by default. Keep all SQL
    # output paths disabled, including forecast-weather snapshot archiving.
    $argsList += "--no-save-sql"
}
if ($SkipDiagnostics) { $argsList += "--skip-diagnostics" }
if ($UseLightGbmGpu) { $argsList += "--use-lgb-gpu" }

$commandText = "`"$PythonExe`" " + (($argsList | ForEach-Object {
    if ($_ -match "\s") { "`"$_`"" } else { $_ }
}) -join " ")

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

function Invoke-GpuPreflight {
    if ($SkipGpuPreflight) {
        Write-Warning "Skipping GPU preflight. The replay may fail later if CUDA is not visible to Python."
        return
    }
    if (!$nvidiaSmi) {
        throw "nvidia-smi was not found. Install the NVIDIA driver or confirm GPU passthrough on the VM before running the Ada replay profile."
    }

    $gpuInfo = & $nvidiaSmi.Source --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -ne 0 -or !$gpuInfo) {
        $gpuText = ($gpuInfo | Out-String).Trim()
        throw "nvidia-smi did not report a usable GPU. Output: $gpuText"
    }
    Write-Host "NVIDIA GPU detected:"
    $gpuInfo | ForEach-Object { Write-Host "  $_" }

    $preflightCode = @'
import json
import os
import sys
import warnings

import numpy as np
import xgboost as xgb

warning_text = ""
X = np.arange(128, dtype=np.float32).reshape(64, 2)
y = X[:, 0] * 0.25 + X[:, 1] * 0.75
model = xgb.XGBRegressor(
    n_estimators=4,
    max_depth=2,
    tree_method="hist",
    device="cuda",
    verbosity=1,
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.fit(X, y)
    warning_text = "\n".join(str(w.message) for w in caught)

booster_config = json.loads(model.get_booster().save_config())
device = ((booster_config.get("learner", {}).get("generic_param", {}) or {}).get("device") or "")
markers = [
    "No visible GPU",
    "Device is changed from GPU to CPU",
    "not compiled with GPU",
    "GPU is not enabled",
]
if str(device).lower().startswith("cpu") or any(m.lower() in warning_text.lower() for m in markers):
    raise SystemExit(
        "XGBoost CUDA preflight failed: trained on device={!r}. Warnings: {}".format(
            device, warning_text[:1000]
        )
    )

try:
    import cupy as cp

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 1:
        raise RuntimeError("CuPy reports zero CUDA devices")
    cp.cuda.Device(0).use()
except Exception as exc:
    raise SystemExit(f"CuPy CUDA preflight failed: {type(exc).__name__}: {exc}")

try:
    from catboost import CatBoostRegressor

    cat = CatBoostRegressor(
        iterations=4,
        depth=2,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices=str(os.environ.get("FORECAST_CUDA_DEVICE", "0")),
        gpu_ram_part=0.10,
        verbose=False,
        allow_writing_files=False,
    )
    cat.fit(X, y)
except Exception as exc:
    raise SystemExit(f"CatBoost GPU preflight failed: {type(exc).__name__}: {exc}")

print(f"GPU preflight ok: xgboost={xgb.__version__}, xgb_device={device}, cupy_devices={device_count}")
'@

    $preflight = & $PythonExe -c $preflightCode 2>&1
    if ($LASTEXITCODE -ne 0) {
        $preflightText = ($preflight | Out-String).Trim()
        throw "Python GPU preflight failed before replay started. $preflightText"
    }
    $preflight | ForEach-Object { Write-Host $_ }
}

Invoke-GpuPreflight

$gpuMonitorJob = $null
if ($nvidiaSmi -and $GpuMonitorIntervalSec -gt 0) {
    "Timestamp,Name,GPU_Util_Pct,Memory_Util_Pct,Memory_Used_MiB,Memory_Total_MiB,Power_Draw_W,Temperature_C" |
        Set-Content -Path $GpuMonitorPath -Encoding UTF8
    $gpuMonitorJob = Start-Job -ScriptBlock {
        param($NvidiaSmiPath, $OutPath, $IntervalSec)
        while ($true) {
            $timestamp = (Get-Date).ToString("o")
            $rows = & $NvidiaSmiPath --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>$null
            foreach ($row in $rows) {
                "$timestamp,$row" | Add-Content -Path $OutPath -Encoding UTF8
            }
            Start-Sleep -Seconds $IntervalSec
        }
    } -ArgumentList $nvidiaSmi.Source, $GpuMonitorPath, $GpuMonitorIntervalSec
}

$startedAt = Get-Date
@{
    run_label = $RunLabel
    status = "running"
    started_at = $startedAt.ToString("o")
    command = $commandText
    config_local = $ServerConfigLocal
    output_dir = $OutputDir
    replay_max_origins = $ReplayMaxOrigins
    replay_processes = $ReplayProcesses
    fixed_origins_file = $FixedOriginsFile
    force_recreate_venv = [bool]$ForceRecreateVenv
    cuda_device = $env:FORECAST_CUDA_DEVICE
    cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
    data_root = $env:FORECAST_DATA_ROOT
    solar_parquet_root = $env:FORECAST_SOLAR_PARQUET_ROOT
    use_lightgbm_gpu = [bool]$UseLightGbmGpu
    save_sql = [bool]$SaveSql
    skip_diagnostics = [bool]$SkipDiagnostics
    skip_gpu_preflight = [bool]$SkipGpuPreflight
    log_path = $LogPath
    summary_path = $SummaryPath
    gpu_monitor_path = $(if ($nvidiaSmi -and $GpuMonitorIntervalSec -gt 0) { $GpuMonitorPath } else { "" })
} | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

Write-Host "Starting Ada server rolling-origin replay: $RunLabel"
Write-Host "Config override: $ServerConfigLocal"
Write-Host "Output: $OutputDir"
Write-Host "Log: $LogPath"
Write-Host "Status: $StatusPath"
if ($nvidiaSmi -and $GpuMonitorIntervalSec -gt 0) {
    Write-Host "GPU monitor: $GpuMonitorPath"
}

try {
    & $PythonExe @argsList *> $LogPath
    $exitCode = $LASTEXITCODE

    $artifacts = @(
        "production_readiness_scorecard.csv",
        "rolling_origin_replay_results.csv",
        "rolling_origin_replay_origin_coverage.csv",
        "rolling_origin_replay_scorecard.csv",
        "rolling_origin_replay_summary.json",
        "rolling_origin_replay_timing.csv",
        "rolling_origin_replay_stage_metrics.csv",
        "rolling_origin_replay_origin_metrics_by_stage.csv",
        "rolling_origin_replay_hot_peak_metrics_by_stage.csv",
        "rolling_origin_replay_hot_peak_candidate_metrics_by_stage.csv",
        "rolling_origin_replay_hot_peak_candidate_scorecard.csv",
        "rolling_origin_replay_hot_ramp_peak_metrics_by_stage.csv",
        "rolling_origin_replay_hot_ramp_peak_candidate_scorecard.csv",
        "rolling_origin_replay_heat_persistence_peak_metrics_by_stage.csv",
        "rolling_origin_replay_heat_persistence_peak_candidate_scorecard.csv",
        "rolling_origin_replay_shoulder_heat_metrics_by_stage.csv",
        "rolling_origin_replay_cloud_solar_midday_metrics_by_stage.csv",
        "rolling_origin_replay_long_horizon_metrics_by_stage.csv",
        "rolling_origin_replay_weather_realism_scorecard.csv",
        "rolling_origin_replay_weather_input_error_by_lead.csv",
        "rolling_origin_replay_weather_input_sensitivity_scorecard.csv",
        "rolling_origin_replay_weather_input_sensitivity_detail.csv",
        "forecast_stage_metrics.csv",
        "feature_importance.csv",
        "model_features.txt",
        "prophet_regressor_features.txt",
        "catboost_features.txt",
        "backtest_metrics.json",
        "backtest_metrics_raw.json",
        "backtest_metrics_final.json",
        "band_coverage_summary.csv",
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

        & $PythonExe scripts\summarize_replay_scorecard.py --output-dir $OutputDir --label $RunLabel *> $SummaryPath

        if ($RunValidation) {
            "`nRunning weather interval validation for $RunLabel..." | Add-Content -Path $LogPath -Encoding UTF8
            & $PythonExe -u scripts\validate_weather_interval_coverage.py `
                --replay-path (Join-Path $OutputDir "rolling_origin_replay_results.csv") `
                --output-label $RunLabel >> $LogPath 2>&1
            $validationExitCode = $LASTEXITCODE
            if ($validationExitCode -ne 0) {
                $exitCode = $validationExitCode
            }
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
        command = $commandText
        config_local = $ServerConfigLocal
        output_dir = $OutputDir
        replay_max_origins = $ReplayMaxOrigins
        replay_processes = $ReplayProcesses
        fixed_origins_file = $FixedOriginsFile
        force_recreate_venv = [bool]$ForceRecreateVenv
        cuda_device = $env:FORECAST_CUDA_DEVICE
        cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
        data_root = $env:FORECAST_DATA_ROOT
        solar_parquet_root = $env:FORECAST_SOLAR_PARQUET_ROOT
        use_lightgbm_gpu = [bool]$UseLightGbmGpu
        save_sql = [bool]$SaveSql
        skip_diagnostics = [bool]$SkipDiagnostics
        run_validation = [bool]$RunValidation
        skip_gpu_preflight = [bool]$SkipGpuPreflight
        log_path = $LogPath
        summary_path = $SummaryPath
        gpu_monitor_path = $(if ($nvidiaSmi -and $GpuMonitorIntervalSec -gt 0) { $GpuMonitorPath } else { "" })
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

    Write-Host "Replay finished with exit code $exitCode"
    Write-Host "Summary: $SummaryPath"
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
        config_local = $ServerConfigLocal
        output_dir = $OutputDir
        replay_max_origins = $ReplayMaxOrigins
        replay_processes = $ReplayProcesses
        force_recreate_venv = [bool]$ForceRecreateVenv
        cuda_device = $env:FORECAST_CUDA_DEVICE
        cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
        data_root = $env:FORECAST_DATA_ROOT
        solar_parquet_root = $env:FORECAST_SOLAR_PARQUET_ROOT
        log_path = $LogPath
        summary_path = $SummaryPath
        gpu_monitor_path = $(if ($nvidiaSmi -and $GpuMonitorIntervalSec -gt 0) { $GpuMonitorPath } else { "" })
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
    throw
}
finally {
    if ($gpuMonitorJob) {
        Stop-Job -Job $gpuMonitorJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job -Job $gpuMonitorJob -ErrorAction SilentlyContinue | Out-Null
    }
}
