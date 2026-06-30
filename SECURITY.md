# Politique de sécurité

## Vulnérabilités connues acceptées

### CVE-2025-3000 — torch.jit.script memory corruption

- **Package concerné** : `torch` (dépendance transitive via `easyocr==1.7.2`)
- **Versions affectées** : `<= 2.12.0`
- **Version corrigée** : aucune disponible à ce jour
- **GHSA** : [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
- **CVSS v4** : `CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N` — **Low**
- **EPSS** : 7e percentile (probabilité d'exploitation très faible)
- **Date d'évaluation** : 2026-06-30

**Décision : risque accepté, alerte dismissée.**

**Justification :**

1. **Code vulnérable non utilisé** : `easyocr==1.7.2` (la seule dépendance introduisant `torch` dans ce projet) n'appelle à aucun moment `torch.jit.script`, `torch.jit.trace` ni `torch.jit.load`. Vérification effectuée directement sur le code source du package (`easyocr/*.py`, `DBNet/`, `model/`).
2. **Aucun usage direct dans le projet** : le code de Rpur n'utilise pas non plus `torch.jit.script` directement.
3. **Vecteur d'attaque local uniquement** (`AV:L`) avec privilèges déjà requis (`PR:L`) : un attaquant devrait déjà avoir un accès local à la machine pour exploiter cette faille, ce qui limite fortement le scénario de risque réel.
4. **Pas de patch disponible** : à ce jour, aucune version corrigée de PyTorch n'existe pour cette vulnérabilité.

**Action de suivi :**

- Revoir cette alerte lors de chaque mise à jour de `torch` ou `easyocr` dans `poetry.lock`.
- Surveiller les [releases PyTorch](https://github.com/pytorch/pytorch/releases) pour un éventuel patch.
- Si `torch.jit.script`/`trace` est introduit dans le projet à l'avenir (directement ou via une nouvelle dépendance), réévaluer immédiatement ce risque.

---

## Signaler une vulnérabilité

Pour signaler une vulnérabilité de sécurité dans ce projet, merci d'ouvrir une issue privée ou de contacter le mainteneur directement plutôt que de la divulguer publiquement.
