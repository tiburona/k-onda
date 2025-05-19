Param(
    [Parameter(Mandatory = $true)]
    [string]$Message
)

$repos = @(
    ".",
    "..\k-onda-analysis"
)

foreach ($repo in $repos) {
    if (Test-Path (Join-Path $repo ".git")) {
        Write-Host "🔄 Committing changes in $repo..."
        Push-Location $repo
        git add -A
        git commit -m $Message
        Pop-Location
    } else {
        Write-Host "⚠️  No Git repo found at $repo — skipping."
    }
}