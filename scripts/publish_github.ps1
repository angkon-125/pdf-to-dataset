# Run only in a CLONED repository after copying in the v0.2 project files.
# This script never force-pushes and ignores secrets and job outputs via .gitignore.
param(
  [string]$RepositoryUrl = 'https://github.com/angkon-125/pdf-to-dataset.git',
  [string]$Branch = 'main'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.git')) {
    throw 'This folder is not a git clone. Clone the GitHub repository before publishing.'
}
$remote = (& git remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'No GitHub origin remote is configured.' }
if ($remote.Trim() -ne $RepositoryUrl) {
    Write-Host "Current origin: $remote"
    throw 'Origin differs from the expected repository. Review the remote before publishing.'
}
& git add -A
if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
$changes = (& git diff --cached --name-only)
if (-not $changes) { Write-Host 'Nothing new to commit.'; exit 0 }
& git commit -m 'Upgrade Angkon PDF-to-Dataset to v0.2'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed. Check git user.name and user.email.' }
& git branch -M $Branch
& git push -u origin $Branch
if ($LASTEXITCODE -ne 0) { throw 'Push failed. Inspect remote changes and use git pull --ff-only or a merge; never force push.' }
Write-Host 'Published successfully.'
