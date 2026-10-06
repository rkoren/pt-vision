# Installs git and ffmpeg with winget and uv with its official installer if missing, creates the project environment.
# have not tested this yet
#   powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
Write-Host "pt-vision setup in $(Get-Location)"

function Have($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }
function RefreshPath {
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
              [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" +
              "$env:USERPROFILE\.local\bin"
}

if (-not (Have "git")) { Write-Host "installing git..."; winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements; RefreshPath }
if (-not (Have "uv"))  { Write-Host "installing uv..."; powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"; RefreshPath }
if (-not (Have "ffmpeg") -or -not (Have "ffprobe")) { Write-Host "installing ffmpeg..."; winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements; RefreshPath }

Write-Host "creating the project environment (first time: a minute or two)..."
uv sync --extra app --extra datasets
Write-Host ""
uv run ptv models pull --mode balanced
uv run ptv version
uv run pre-commit install
Write-Host ""
Write-Host "Next:  uv run ptv demo        # opens the viewer on the bundled sample clip"
Write-Host "       uv run ptv app         # then drop your own video into the window"
Write-Host "If a command is still 'not recognized', open a new terminal and run it again."
