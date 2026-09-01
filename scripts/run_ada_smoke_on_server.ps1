param(
    [string]$RunLabel = "ada_smoke",
    [int]$ReplayMaxOrigins = 2,
    [string]$ServerConfigLocal = "",
    [string]$OutputDir = "",
    [string]$CudaDevice = "0",
    [string]$PythonExe = "",
    [int]$ReplayProcesses = 3,
    [string]$FixedOriginsFile = "",
    [switch]$UpdateEnvironment,
    [switch]$ReuseExistingVenv,
    [switch]$SkipBootstrap,
    [switch]$SkipSolarForecast,
    [switch]$AllowStaleSolarForecast,
    [switch]$SaveSql,
    [switch]$UseLightGbmGpu,
    [switch]$SkipGpuPreflight
)

$ErrorActionPreference = "Stop"

$runnerArgs = @{
    RunLabel = $RunLabel
    ReplayMaxOrigins = $ReplayMaxOrigins
    ReplayProcesses = $ReplayProcesses
    CudaDevice = $CudaDevice
    SkipDiagnostics = $true
}

if (![string]::IsNullOrWhiteSpace($ServerConfigLocal)) {
    $runnerArgs.ServerConfigLocal = $ServerConfigLocal
}
if (![string]::IsNullOrWhiteSpace($OutputDir)) {
    $runnerArgs.OutputDir = $OutputDir
}
if (![string]::IsNullOrWhiteSpace($PythonExe)) {
    $runnerArgs.PythonExe = $PythonExe
}
if (![string]::IsNullOrWhiteSpace($FixedOriginsFile)) {
    $runnerArgs.FixedOriginsFile = $FixedOriginsFile
}
if ($UpdateEnvironment) {
    $runnerArgs.UpdateEnvironment = $true
}
if (!$ReuseExistingVenv) {
    $runnerArgs.ForceRecreateVenv = $true
}
if ($SkipBootstrap) {
    $runnerArgs.SkipBootstrap = $true
}
if ($SkipSolarForecast) {
    $runnerArgs.SkipSolarForecast = $true
}
if ($AllowStaleSolarForecast) {
    $runnerArgs.AllowStaleSolarForecast = $true
}
if ($SaveSql) {
    $runnerArgs.SaveSql = $true
}
else {
    $runnerArgs.NoSaveSql = $true
}
if ($UseLightGbmGpu) {
    $runnerArgs.UseLightGbmGpu = $true
}
if ($SkipGpuPreflight) {
    $runnerArgs.SkipGpuPreflight = $true
}

& (Join-Path $PSScriptRoot "run_rolling_origin_replay_ada_server.ps1") @runnerArgs
exit $LASTEXITCODE
