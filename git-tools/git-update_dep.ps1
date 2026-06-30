<# 
Utilisation :
.\git-tools\git-update_dep.ps1 -package <nom_du_package>
ex : .\git-tools\git-update_dep.ps1 -package pandas
.\git-tools\git-update_dep.ps1 -package <nom_du_package> -branchPrefix <prefix_de_branche>
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$package,

    [string]$branchPrefix = "deps/update"
)

$ErrorActionPreference = "Stop"

# Se place à la racine du dépôt pour fiabiliser les chemins après le déplacement
# du script dans le sous-dossier git-tools.
function Get-RepoRoot {
    $repoRoot = git rev-parse --show-toplevel 2>$null
    if ($repoRoot) {
        return $repoRoot.Trim()
    }

    return (Split-Path -Parent $PSScriptRoot)
}

# Récupère la version actuellement installée d'une dépendance Poetry.
function Get-PackageVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    poetry show $Name 2>$null |
        Select-String -Pattern "(\d+\.\d+\.\d+)" |
        ForEach-Object { $_.Matches[0].Value } |
        Select-Object -First 1
}

# Lit le seuil minimal de couverture défini dans pyproject.toml.
function Get-CoverageThreshold {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyprojectPath
    )

    $content = Get-Content -Path $PyprojectPath
    $inCoverageReportSection = $false

    foreach ($line in $content) {
        if ($line -match '^\[tool\.coverage\.report\]') {
            $inCoverageReportSection = $true
            continue
        }

        if ($inCoverageReportSection -and $line -match '^\[') {
            break
        }

        if ($inCoverageReportSection -and $line -match '^\s*fail_under\s*=\s*(\d+)') {
            return [int]$matches[1]
        }

        if ($line -match '--cov-fail-under=(\d+)') {
            return [int]$matches[1]
        }
    }

    throw "Impossible de lire le seuil de couverture dans pyproject.toml."
}

if ($package.StartsWith("-")) {
    $package = $package.TrimStart("-")
}

Write-Host "📦 Mise à jour de $package"

$repoRoot = Get-RepoRoot
Set-Location $repoRoot

# Définit le chemin du pyproject pour relire la configuration de couverture.
$pyprojectPath = Join-Path $repoRoot "pyproject.toml"
$coverageThreshold = Get-CoverageThreshold -PyprojectPath $pyprojectPath

# Localise l'environnement virtuel géré par Poetry.
$poetryEnvPath = poetry env info --path 2>$null
if (-not $poetryEnvPath) {
    Write-Host "❌ Impossible de trouver l'environnement Poetry."
    exit 1
}

# Vérifie que le script d'activation PowerShell existe bien.
$activateScript = Join-Path $poetryEnvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "❌ Script d'activation introuvable: $activateScript"
    exit 1
}

# Active l'environnement Poetry pour exécuter les commandes dans le bon contexte.
. $activateScript

# Lit la version actuelle du package avant la mise à jour.
$currentVersion = Get-PackageVersion -Name $package
if (-not $currentVersion) {
    Write-Host "❌ Package $package non trouvé"
    exit 1
}

Write-Host "Version actuelle : $currentVersion"
Write-Host "⬆️ Mise à jour vers la dernière version..."

# Met à jour la dépendance vers sa dernière version disponible.
poetry add "$package@latest"

# Relit la version du package après la mise à jour.
$newVersion = Get-PackageVersion -Name $package
Write-Host "Nouvelle version : $newVersion"

# Arrête le script si aucune mise à jour n'a été appliquée.
if ($currentVersion -eq $newVersion) {
    Write-Host "✅ Déjà à jour"
    exit 0
}

Write-Host "✅ $package : $currentVersion -> $newVersion"
Write-Host "🧪 Lancement des tests pytest avec couverture minimale à $coverageThreshold%"

# Lance les tests et impose le seuil de couverture défini dans pyproject.toml.
poetry run pytest "--cov-fail-under=$coverageThreshold"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Tests échoués. Annulation."
    git checkout pyproject.toml poetry.lock
    exit 1
}

Write-Host "✅ Tests OK"

# Crée une branche dédiée à la mise à jour de dépendance.
$branchName = "$branchPrefix-$package-$newVersion"
git checkout -b $branchName

# Commit et pousse les fichiers modifiés vers le dépôt distant.
git add pyproject.toml poetry.lock
git commit -m "⬆️ bump $package from $currentVersion to $newVersion"
git push origin $branchName

# Ouvre automatiquement une pull request via GitHub CLI.
gh pr create `
  --title "⬆️ Update $package to $newVersion" `
  --body "Automated update + tests OK (`pytest`)." `
  --base main `
  --head $branchName

Write-Host "🚀 PR créée"
