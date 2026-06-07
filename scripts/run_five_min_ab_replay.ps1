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

$OnLabel = "fivemin_on_$RunStamp"
$OffLabel = "fivemin_off_$RunStamp"
$ReplayScript = Join-Path $PSScriptRoot "run_full_replay_overnight.ps1"
$OutputDir = Join-Path $RepoRoot "forecast_outputs"
$GeneratedOriginsFile = Join-Path $OutputDir "fixed_replay_origins_$RunStamp.txt"

Write-Host "Starting five-minute ON replay: $OnLabel"
if ([string]::IsNullOrWhiteSpace($FixedOriginsFile)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ReplayScript -RunLabel $OnLabel
} else {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ReplayScript -RunLabel $OnLabel -FixedOriginsFile $FixedOriginsFile
}
if ($LASTEXITCODE -ne 0) {
    throw "Five-minute ON replay failed with exit code $LASTEXITCODE"
}

$OriginsForOff = $FixedOriginsFile
if ([string]::IsNullOrWhiteSpace($OriginsForOff)) {
    $OnReplayResults = Join-Path $OutputDir "rolling_origin_replay_results_$OnLabel.csv"
    if (!(Test-Path $OnReplayResults)) {
        throw "Missing labeled ON replay results; cannot extract fixed origins for OFF replay."
    }
    Import-Csv $OnReplayResults |
        Select-Object -ExpandProperty Replay_Origin_DT -Unique |
        Set-Content -Path $GeneratedOriginsFile -Encoding ASCII
    $OriginsForOff = $GeneratedOriginsFile
    Write-Host "Saved fixed replay origins for OFF leg to: $GeneratedOriginsFile"
}

Write-Host "Starting five-minute OFF replay: $OffLabel"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ReplayScript -RunLabel $OffLabel -DisableFiveMinLoad -FixedOriginsFile $OriginsForOff
if ($LASTEXITCODE -ne 0) {
    throw "Five-minute OFF replay failed with exit code $LASTEXITCODE"
}

$OnScorecard = Join-Path $OutputDir "production_readiness_scorecard_$OnLabel.csv"
$OffScorecard = Join-Path $OutputDir "production_readiness_scorecard_$OffLabel.csv"
if (!(Test-Path $OnScorecard) -or !(Test-Path $OffScorecard)) {
    throw "Missing labeled production readiness scorecards for comparison."
}

$on = Import-Csv $OnScorecard
$off = Import-Csv $OffScorecard
$comparison = foreach ($row in $on) {
    $match = $off | Where-Object { $_.Test -eq $row.Test } | Select-Object -First 1
    if ($null -eq $match) {
        continue
    }
    [pscustomobject]@{
        Test = $row.Test
        FiveMinOn_MAE = [math]::Round([double]$row.MAE_MWH, 4)
        FiveMinOff_MAE = [math]::Round([double]$match.MAE_MWH, 4)
        Delta_MAE_OnMinusOff = [math]::Round(([double]$row.MAE_MWH - [double]$match.MAE_MWH), 4)
        FiveMinOn_MAPE = [math]::Round([double]$row.MAPE_PCT, 4)
        FiveMinOff_MAPE = [math]::Round([double]$match.MAPE_PCT, 4)
        Delta_MAPE_OnMinusOff = [math]::Round(([double]$row.MAPE_PCT - [double]$match.MAPE_PCT), 4)
        FiveMinOn_Bias = [math]::Round([double]$row.Bias_MWH, 4)
        FiveMinOff_Bias = [math]::Round([double]$match.Bias_MWH, 4)
        FiveMinOn_Pass = $row.Pass
        FiveMinOff_Pass = $match.Pass
    }
}

$ComparisonPath = Join-Path $OutputDir "five_min_ab_replay_comparison_$RunStamp.csv"
$comparison | Export-Csv -Path $ComparisonPath -NoTypeInformation
$comparison | Format-Table -AutoSize
Write-Host "Saved A/B comparison to: $ComparisonPath"
