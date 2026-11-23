# Clone or safely update the Angkon GitHub repository from the extracted v0.3 ZIP.
# Run on YOUR Windows computer; does not force push or include datasets/private PDFs.
param(
    [string]$BaseDir = 'E:\DOWNLOADS',
    [string]$RepositoryUrl = 'https://github.com/angkon-125/pdf-to-dataset.git'
)
$ErrorActionPreference = 'Stop'
$source = Join-Path $BaseDir 'angkon_dataset_factory'
$target = Join-Path $BaseDir 'pdf-to-dataset'
if (-not (Test-Path (Join-Path $source 'curator_app.py'))) { throw "Project v0.3 not found: $source" }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows first.' }

if (Test-Path (Join-Path $target '.git')) {
    $remote = (& git -C $target remote get-url origin)
    if ($LASTEXITCODE -ne 0 -or $remote.Trim() -ne $RepositoryUrl) {
        throw "Git remote mismatch: $remote"
    }
    $dirty = (& git -C $target status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw 'git status failed.' }
    if ($dirty) { throw 'Git repository has uncommitted changes. Commit/stash/review them before updating.' }
    & git -C $target pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Could not fast-forward. Resolve branch changes; do not force-push.' }
} else {
    if (Test-Path $target) { throw "Target folder exists but is not a Git repository: $target" }
    & git clone $RepositoryUrl $target
    if ($LASTEXITCODE -ne 0) { throw 'Git clone failed.' }
}

& robocopy $source $target /E /XD '.git' '.venv' 'venv' 'runs' 'uploads' 'private' 'local_data' '__pycache__' '.pytest_cache' '.ruff_cache' 'review_workbench' 'curated_exports' /XF '*.zip' '*.csv' '*.jsonl' '*.pdf' '*.xlsx' '*.parquet' '*.gguf' '*.pt' '*.pth' '*.ckpt' '*.safetensors' '.env' '.env.local' '*.tmp' '*.log'
$copyExit = $LASTEXITCODE
if ($copyExit -gt 7) { throw "Robocopy error code $copyExit" }
# Safe, intentionally synthetic demo PDF from the packaged project.
$demo = Join-Path $source 'examples\demo.pdf'
if (Test-Path $demo) {
    $examples = Join-Path $target 'examples'
    New-Item -Path $examples -ItemType Directory -Force | Out-Null
    Copy-Item $demo (Join-Path $examples 'demo.pdf') -Force
}

& git -C $target add -A
if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
$staged = (& git -C $target diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Could not check staged files.' }
$blocked = @($staged | Where-Object {
    $_ -match '(?i)(^|/)(\.env(\.|$)|runs/|uploads/|private/|review_workbench/|local_data/)' -or
    ($_ -match '(?i)\.(pdf|csv|jsonl|xlsx|parquet|zip|pt|pth|gguf|ckpt|safetensors)$' -and $_ -ne 'examples/demo.pdf')
})
if ($blocked.Count -gt 0) {
    Write-Host 'Blocked potentially sensitive files:'
    $blocked | ForEach-Object { Write-Host "  $_" }
    throw 'Review/remove blocked files from the index before publishing. No commit was created.'
}
if (-not $staged) { Write-Host 'No project changes to publish.'; exit 0 }
Write-Host 'Reviewing changes to upload:'
$staged | ForEach-Object { Write-Host "  $_" }
& git -C $target commit -m 'Upgrade Angkon Dataset Factory to v0.3 - Human Annotator and Data Science Lab'
if ($LASTEXITCODE -ne 0) { throw 'Git commit failed. Check Git user identity.' }
& git -C $target branch -M main
& git -C $target push -u origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed. Review remote state; do not force push.' }
Write-Host "Updated: $RepositoryUrl"
