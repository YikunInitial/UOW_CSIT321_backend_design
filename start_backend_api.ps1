param(
    [int]$Port = 8080,
    [string]$DataRoot = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found. Install requirements before starting the API."
}
if ($DataRoot) {
    $resolved = [System.IO.Path]::GetFullPath($DataRoot)
    New-Item -ItemType Directory -Force -Path $resolved | Out-Null
    $env:STEEL_API_RUN_ROOT = $resolved
    $env:STEEL_API_STATE_DATABASE = Join-Path $resolved "state.db"
}
& $python -m uvicorn backend_api:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE

