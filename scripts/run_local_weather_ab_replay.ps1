param(
    [string]$RunStamp = "",
    [string]$FixedOriginsFile = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($RunStamp)) {
    $RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
}

$BaseLabel = "localwx_base_$RunStamp"
$CalLabel = "localwx_cal_$RunStamp"
$ReplayScript = Join-Path $PSScriptRoot "run_full_replay_overnight.ps1"
$OutputDir = Join-Path $RepoRoot "forecast_outputs"
$GeneratedOriginsFile = Join-Path $OutputDir "fixed_replay_origins_localwx_$RunStamp.txt"

Write-Host "Starting Open-Meteo baseline replay: $BaseLabel"
if ([string]::IsNullOrWhiteSpace($FixedOriginsFile)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ReplayScript -RunLabel $BaseLabel
} else {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ReplayScript -RunLabel $BaseLabel -FixedOriginsFile $FixedOriginsFile
}
if ($LASTEXITCODE -ne 0) {
    throw "Open-Meteo baseline replay failed with exit code $LASTEXITCODE"
}

$OriginsForCal = $FixedOriginsFile
if ([string]::IsNullOrWhiteSpace($OriginsForCal)) {
    $BaseReplayResults = Join-Path $OutputDir "rolling_origin_replay_results_$BaseLabel.csv"
    if (!(Test-Path $BaseReplayResults)) {
        throw "Missing labeled baseline replay results; cannot extract fixed origins for calibrated replay."
    }
    Import-Csv $BaseReplayResults |
        Select-Object -ExpandProperty Replay_Origin_DT -Unique |
        Set-Content -Path $GeneratedOriginsFile -Encoding ASCII
    $OriginsForCal = $GeneratedOriginsFile
    Write-Host "Saved fixed replay origins for calibrated leg to: $GeneratedOriginsFile"
}

Write-Host "Starting Berry-calibrated weather replay: $CalLabel"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ReplayScript -RunLabel $CalLabel -UseLocalWeatherCalibration -FixedOriginsFile $OriginsForCal
if ($LASTEXITCODE -ne 0) {
    throw "Berry-calibrated weather replay failed with exit code $LASTEXITCODE"
}

$BaseScorecard = Join-Path $OutputDir "production_readiness_scorecard_$BaseLabel.csv"
$CalScorecard = Join-Path $OutputDir "production_readiness_scorecard_$CalLabel.csv"
if (!(Test-Path $BaseScorecard) -or !(Test-Path $CalScorecard)) {
    throw "Missing labeled production readiness scorecards for comparison."
}

$base = Import-Csv $BaseScorecard
$cal = Import-Csv $CalScorecard
$comparison = foreach ($row in $cal) {
    $match = $base | Where-Object { $_.Test -eq $row.Test } | Select-Object -First 1
    if ($null -eq $match) {
        continue
    }
    [pscustomobject]@{
        Test = $row.Test
        BerryCal_MAE = [math]::Round([double]$row.MAE_MWH, 4)
        OpenMeteoBase_MAE = [math]::Round([double]$match.MAE_MWH, 4)
        Delta_MAE_CalMinusBase = [math]::Round(([double]$row.MAE_MWH - [double]$match.MAE_MWH), 4)
        BerryCal_MAPE = [math]::Round([double]$row.MAPE_PCT, 4)
        OpenMeteoBase_MAPE = [math]::Round([double]$match.MAPE_PCT, 4)
        Delta_MAPE_CalMinusBase = [math]::Round(([double]$row.MAPE_PCT - [double]$match.MAPE_PCT), 4)
        BerryCal_Bias = [math]::Round([double]$row.Bias_MWH, 4)
        OpenMeteoBase_Bias = [math]::Round([double]$match.Bias_MWH, 4)
        BerryCal_Pass = $row.Pass
        OpenMeteoBase_Pass = $match.Pass
    }
}

$ComparisonPath = Join-Path $OutputDir "local_weather_ab_replay_comparison_$RunStamp.csv"
$comparison | Export-Csv -Path $ComparisonPath -NoTypeInformation
$comparison | Format-Table -AutoSize
Write-Host "Saved local weather A/B comparison to: $ComparisonPath"
