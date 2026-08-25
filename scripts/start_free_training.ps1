$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$notebook = Join-Path $repoRoot "notebooks\free_first_bdx_walk.ipynb"
$guide = Join-Path $repoRoot "docs\FREE_FIRST_BDX_POLICY.md"
$branch = "codex/free-first-bdx-policy"
$forkUrl = "https://github.com/l2smith2/Open_Duck_Playground.git"
$kaggleNewNotebook = "https://www.kaggle.com/code/new"

Set-Location $repoRoot
Write-Host "Open Duck free-first setup" -ForegroundColor Cyan
Write-Host "Repository: $repoRoot"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or is not available in PATH."
}
if (-not (Test-Path -LiteralPath $notebook)) {
    throw "Training notebook was not found: $notebook"
}

# ConvertFrom-Json catches a damaged notebook before the browser is opened.
$null = Get-Content -LiteralPath $notebook -Raw | ConvertFrom-Json

$currentBranch = (git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "This folder is not a usable Git repository."
}
if ($currentBranch -ne $branch) {
    throw "Expected branch $branch but found $currentBranch."
}

$origin = (git remote get-url origin 2>$null).Trim()
if ($origin -ne $forkUrl) {
    Write-Host "Correcting origin to your fork..."
    git remote set-url origin $forkUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Could not configure the GitHub fork as origin."
    }
}

$changes = @(git status --porcelain)
if ($changes.Count -gt 0) {
    Write-Host ""
    Write-Host "There are unpublished local changes:" -ForegroundColor Yellow
    $changes | ForEach-Object { Write-Host "  $_" }
    $answer = Read-Host "Commit and publish these changes now? Type y to continue"
    if ($answer -notin @("y", "Y", "yes", "YES")) {
        throw "Nothing was published. Run this launcher again when ready."
    }
    git add --all
    if ($LASTEXITCODE -ne 0) { throw "git add failed." }
    git commit -m "Update free-first Open Duck training workflow"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
}

Write-Host "Publishing $branch to your GitHub fork..."
git push --set-upstream origin $branch
if ($LASTEXITCODE -ne 0) {
    throw "Git push failed. Git Credential Manager may need you to sign in."
}

Set-Clipboard -Value $notebook
Write-Host ""
Write-Host "GitHub is ready and the notebook is valid." -ForegroundColor Green
Write-Host "Kaggle will open now. The notebook file will be highlighted in Explorer."
Write-Host ""
Write-Host "In Kaggle:" -ForegroundColor Cyan
Write-Host "  1. Sign in if asked."
Write-Host "  2. Choose File > Import Notebook > Upload."
Write-Host "  3. Select the highlighted free_first_bdx_walk.ipynb file."
Write-Host "  4. Enable Internet and a P100 GPU (T4 if P100 is unavailable)."
Write-Host "  5. Run one section at a time, starting with Setup."
Write-Host ""
Write-Host "The notebook path is also on your clipboard as a fallback."

$explorerArgument = '/select,"' + $notebook + '"'
Start-Process $kaggleNewNotebook
Start-Process explorer.exe -ArgumentList $explorerArgument

Write-Host ""
Write-Host "Beginner guide: $guide"
