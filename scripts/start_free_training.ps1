$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$templateNotebook = Join-Path $repoRoot "notebooks\free_first_bdx_walk.ipynb"
$localKaggleDir = Join-Path $repoRoot ".local-kaggle"
$notebook = Join-Path $localKaggleDir "free_first_bdx_walk.ipynb"
$guide = Join-Path $repoRoot "docs\FREE_FIRST_BDX_POLICY.md"
$branch = "codex/free-first-bdx-policy"
$kaggleNewNotebook = "https://www.kaggle.com/code/new"

Set-Location $repoRoot
Write-Host "Open Duck free-first setup" -ForegroundColor Cyan
Write-Host "Repository: $repoRoot"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or is not available in PATH."
}
if (-not (Test-Path -LiteralPath $templateNotebook)) {
    throw "Training notebook template was not found: $templateNotebook"
}

# ConvertFrom-Json catches a damaged notebook before the browser is opened.
$templateText = Get-Content -LiteralPath $templateNotebook -Raw
$null = $templateText | ConvertFrom-Json

$currentBranch = (git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "This folder is not a usable Git repository."
}
if ($currentBranch -ne $branch) {
    throw "Expected branch $branch but found $currentBranch."
}

$origin = (git remote get-url origin 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or -not $origin) {
    throw "No Git origin is configured. Set origin to your GitHub fork first."
}
if ($origin -match '^https://github\.com/([^/]+/Open_Duck_Playground(?:\.git)?)$') {
    $forkUrl = $origin
}
elseif ($origin -match '^git@github\.com:([^/]+/Open_Duck_Playground(?:\.git)?)$') {
    $forkUrl = "https://github.com/$($Matches[1])"
}
else {
    throw "Git origin must be your GitHub Open_Duck_Playground fork."
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

# Put the account-specific URL only in a local file that .gitignore excludes.
$placeholder = "https://github.com/YOUR_GITHUB_USER/Open_Duck_Playground.git"
if (-not $templateText.Contains($placeholder)) {
    throw "The notebook template does not contain the expected generic fork placeholder."
}
New-Item -ItemType Directory -Path $localKaggleDir -Force | Out-Null
$configuredText = $templateText.Replace($placeholder, $forkUrl)
[System.IO.File]::WriteAllText(
    $notebook,
    $configuredText,
    [System.Text.UTF8Encoding]::new($false)
)
$null = Get-Content -LiteralPath $notebook -Raw | ConvertFrom-Json

Set-Clipboard -Value $notebook
Write-Host ""
Write-Host "GitHub is ready. A local Kaggle upload copy has been created." -ForegroundColor Green
Write-Host "Kaggle will open now and Explorer will highlight the correct file."
Write-Host ""
Write-Host "In Kaggle:" -ForegroundColor Cyan
Write-Host "  1. Sign in if asked."
Write-Host "  2. If you see Import Notebook, choose Local. In the editor use File > Import Notebook."
Write-Host "  3. Select the highlighted .local-kaggle\free_first_bdx_walk.ipynb file."
Write-Host "  4. If asked, choose Quick Save - do not choose Commit & Run or Save & Run All."
Write-Host "  5. In the right Settings pane, turn Internet on."
Write-Host "  6. Turn File persistence on."
Write-Host "  7. Under Accelerator choose T4 x2 (P100 is acceptable). Confirm the restart."
Write-Host "  8. Run only the first Setup code cell and check that it prints Ready: gpu."
Write-Host ""
Write-Host "The notebook path is also on your clipboard as a fallback."

$explorerArgument = '/select,"' + $notebook + '"'
Start-Process $kaggleNewNotebook
Start-Process explorer.exe -ArgumentList $explorerArgument

Write-Host ""
Write-Host "Beginner guide: $guide"
