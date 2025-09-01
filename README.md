# Rpur OCR - Utilitaires et Script Principal

Ce projet est une solution d’ETL (Extract, Transform, Load) spécialisée en OCR (reconnaissance optique de caractères) : il permet d’extraire du texte à partir d’images, de transformer et structurer ces données, puis de générer des rapports Word et d’alimenter une base de données pour le suivi d’inspections de cheminées.

## Auteur

- Pierrick BERTHE  
- pierrick.berthe@gmx.fr  
- Août 2025

---

## Fonctionnalités principales

- **Extraction OCR** : Utilisation d’EasyOCR pour extraire du texte sur images.
- **Prétraitement d’images** : Conversion, binarisation, redimensionnement, compression.
- **Génération de rapports Word** : Insertion d’images, de textes, de remarques, de logos, et de pied de page personnalisés.
- **Gestion de base de données SQLite** : Création automatique des tables, insertion et suivi des clients, cheminées et mesures.
- **Journalisation** : Redirection de tous les prints dans un fichier log.
- **Gestion des dossiers et fichiers temporaires**.

---

## Technologies et librairies utilisées

- **Python 3.12**
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [python-docx](https://python-docx.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [language-tool-python](https://github.com/jxmorris12/language-tool-python)
- [tqdm](https://tqdm.github.io/)
- [sqlite3](https://docs.python.org/3/library/sqlite3.html)

---

## Installation

### Prérequis

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- Git (pour cloner le dépôt)

### Installation des dépendances

```sh
git clone https://github.com/ton-utilisateur/ton-repo.git
cd ton-repo
poetry install
```

---

## Utilisation

### Lancement du script principal

```sh
poetry run python src/main.py
```
ou via le fichier batch :
```bat
Rpur_extractor.bat
```

### Fonctionnement général

1. **Saisie utilisateur** : initiales client, date de mesure, dossier à ignorer.
2. **Extraction de texte** : OCR sur les images du dossier source, prétraitement et redimensionnement.
3. **Génération de rapports** : création d’un rapport Word avec photos, remarques, logo, pied de page et pagination.
4. **Mise à jour base de données** : insertion des clients, cheminées, mesures (avec gestion des doublons).
5. **Nettoyage** : suppression des fichiers temporaires.

---

## Arborescence attendue

```
Rpur/
├── bdd/                   # Base de données SQLite
├── data/
│   ├── input/source/      # Images à traiter
│   └── output/            # Résultats du traitement
├── doc/                   # Documentation
├── image/                 # Logo
├── notebook/              # Notebooks de POC
├── src/
│   ├── config.py          # Configuration des paramètres
│   ├── main.py            # Script principal
│   └── ocr_utils.py       # Fonctions utilitaires pour l'OCR
├── .gitignore              # Fichiers et dossiers à ignorer par Git
├── poetry.lock             # Verrouillage des dépendances
├── pyproject.toml
├──  README.md
├──  Rpur_extractor.bat     # Script de lancement
```

---

## Notes techniques

- **Gestion des erreurs** : Toutes les fonctions critiques (I/O, DB, image) sont protégées par des try/except avec messages explicites.
- **OCR** : EasyOCR est utilisé en mode CPU (`gpu=False`), batch configurable.
- **Images** : Les images sont prétraitées (binarisation, redimensionnement, compression) avant insertion dans Word.
- **Word** : Le rapport Word est généré avec insertion dynamique des images, remarques, titres, logo, pied de page et pagination automatique.
- **Base de données** : Création automatique des tables, insertion avec gestion des doublons, comptage des ajouts.
- **Journalisation** : Tous les prints sont redirigés dans `data/output/log/process_log.txt`.
- **Extensible** : Les fonctions sont modulaires et facilement réutilisables.

---

## Modélisation de la base de donnée

┌───────────────────┐
│     clients       │
├───────────────────┤
│ client_id   (PK)  │
│ nom               │
└───────────────────┘
          │ 1
          │
          │ n
          ▼
┌────────────────────────────────────────┐
│               cheminees                │
├────────────────────────────────────────┤
│ cheminee_id   (PK)                     │
│ client_id     (FK → clients.client_id) │
│ localisation  (PK)                     │
│ remarques                              │
└────────────────────────────────────────┘
          │ 1
          │
          │ n
          ▼
┌───────────────────────────────────────────────┐
│                  mesures                      │
├───────────────────────────────────────────────┤
│ mesure_id    (PK)                             │
│ client_id   (FK → clients.client_id)          │
│ cheminee_id  (FK → cheminees.cheminee_id)     │
│ date_mesure                                   │
└───────────────────────────────────────────────┘

---

## Contact

Pour toute question ou amélioration, contacter :  
**Pierrick BERTHE** – pierrick.berthe@gmx.fr
