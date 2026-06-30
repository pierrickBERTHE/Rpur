<#
Utilisation :
.\git-tools\git-update_flag_version.ps1

Ce script :
1. Récupère le dernier tag Git
2. Te demande de saisir le nouveau tag
3. Te demande un message de commit personnalisé (ou utilise le défaut)
4. Crée le nouveau tag annoté avec le message
5. Te propose de pousser le tag
#>

$ErrorActionPreference = "Stop"

# Se place à la racine du dépôt pour que les commandes git s'exécutent
# correctement depuis le sous-dossier git-tools.
$repoRoot = git rev-parse --show-toplevel 2>$null
if (-not $repoRoot) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
}
Set-Location $repoRoot.Trim()

Write-Host "📌 Script de mise à jour de version (Git Tag)"

# Récupère le dernier tag
$latestTag = git describe --tags --abbrev=0 2>$null

if ($latestTag) {
    Write-Host "✅ Dernier tag trouvé: $latestTag"
} else {
    Write-Host "⚠️  Aucun tag trouvé dans ce repository"
    $latestTag = "v0.0.0"
    Write-Host "   Utilisation de valeur par défaut: $latestTag"
}

# Demande à l'utilisateur le nouveau tag
Write-Host ""
Write-Host "📝 Saisir le nouveau tag (ex: v1.2.3)"
Write-Host "   Dernier: $latestTag"
Write-Host ""
$newTag = Read-Host "Nouveau tag"

# Validation du format
if ($newTag -notmatch '^v\d+\.\d+\.\d+') {
    Write-Host "❌ Format invalide. Utilise le format: v1.2.3"
    exit 1
}

# Demande le message de commit
Write-Host ""
Write-Host "📝 Message de commit (optionnel, appuie sur Entrée pour ignorer)"
$commitMessage = Read-Host "Message"

if (-not $commitMessage) {
    $commitMessage = "Release $newTag"
}

Write-Host ""
Write-Host "🔄 Création du tag: $newTag"
Write-Host "   Message: $commitMessage"

# Crée et pousse le tag
try {
    git tag -a $newTag -m $commitMessage
    Write-Host "✅ Tag créé localement: $newTag"
    
    # Propose de pousser le tag
    Write-Host ""
    $pushChoice = Read-Host "Pousser vers le repository distant ? (o/n)"
    
    if ($pushChoice -eq "o" -or $pushChoice -eq "oui") {
        git push origin $newTag
        Write-Host "✅ Tag poussé: $newTag"
    } else {
        Write-Host "ℹ️  Tag local créé. À pousser manuellement avec:"
        Write-Host "   git push origin $newTag"
    }
    
    Write-Host ""
    Write-Host "🚀 Succès!"
    
} catch {
    Write-Host "❌ Erreur: $_"
    exit 1
}
