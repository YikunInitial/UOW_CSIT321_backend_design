param(
    [string]$DataRoot = "",
    [double]$LeaseSeconds = 90,
    [double]$PollSeconds = 1
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found. Install requirements before starting the worker."
}
if ($DataRoot) {
    $resolved = [System.IO.Path]::GetFullPath($DataRoot)
    New-Item -ItemType Directory -Force -Path $resolved | Out-Null
    $env:STEEL_API_RUN_ROOT = $resolved
    $env:STEEL_API_STATE_DATABASE = Join-Path $resolved "state.db"
}
& $python worker.py `
    --database $env:STEEL_API_STATE_DATABASE `
    --lease-seconds $LeaseSeconds `
    --poll-seconds $PollSeconds
exit $LASTEXITCODE
