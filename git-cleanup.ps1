# Basculer sur la branche main
git checkout main

# Récupérer les dernières modifications depuis le dépôt distant
git pull origin main

# Lister toutes les branches déjà fusionnées dans main (sauf main elle-même)
$merged = git branch --merged main | Where-Object { $_ -notmatch "main" }

# Parcourir chaque branche fusionnée
foreach ($branch in $merged) {
    # Nettoyer le nom de la branche (supprimer les espaces)
    $branchName = $branch.Trim()
    
    # Supprimer la branche locale
    git branch -d $branchName
}

# Nettoyer les références aux branches distantes supprimées
git remote prune origin

# Afficher un message de confirmation
Write-Host "✅ Nettoyage terminé !"