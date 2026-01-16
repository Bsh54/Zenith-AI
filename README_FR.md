# Zenith AI 
> **Analyse Vidéo Multimodale Haute Performance & Synthèse Narrative**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Bsh54/Zenith-AI/blob/main/main.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 🌐 Langue / Language
**Français** | [🇺🇸 View the README in English](./README.md)

---

## ✨ Présentation
**Zenith AI** est un système d'intelligence multimodale de pointe conçu pour "comprendre" le contenu vidéo comme un être humain. En combinant la vision par ordinateur (YOLOv8), la transcription audio (Whisper) et les grands modèles de langage (LLM), il transforme n'importe quelle vidéo ou URL en un rapport narratif structuré et professionnel.

### 🚀 Fonctionnalités Clés
- **🎥 Entrée Universelle** : Chargez des fichiers locaux ou collez des liens (YouTube, TikTok, Twitter, etc.).
- **👁️ Intelligence Visuelle** : Détection d'objets et analyse de scène en temps réel avec YOLOv8.
- **🎙️ Transcription Audio** : Transcription haute fidélité avec détection automatique de la langue.
- **🧠 Synthèse Narrative** : Génère un rapport d'analyse contextuel approfondi en français.
- **💎 Interface de Luxe** : Un tableau de bord moderne en mode sombre conçu avec Gradio.

---

## 🛠️ Comment exécuter sur Google Colab

Suivez ces étapes simples pour lancer Zenith AI en quelques secondes :

### 1. Ouvrir un nouveau Notebook
Allez sur [Google Colab](https://colab.research.google.com/) et créez un nouveau notebook Python 3.

### 2. Configurer l'accélération GPU (Recommandé)
Pour des performances maximales :
- Allez dans `Exécution` > `Modifier le type d'exécution`
- Sélectionnez **T4 GPU** (ou tout GPU disponible)
- Cliquez sur **Enregistrer**

### 3. Copier et Coller le Code
Copiez l'intégralité du contenu de [main.ipynb](./main.ipynb) dans une cellule.

### 4. Configurer votre API
Avant de lancer la cellule, trouvez la section `API_CONFIG` en haut du script et entrez vos accès :
```python
API_CONFIG = {
    "url": "VOTRE_ENDPOINT_API",
    "key": "VOTRE_CLE_API",
    "model": "VOTRE_NOM_DE_MODELE"
}
```

### 5. Lancer l'application
- Exécutez la cellule (Ctrl + Entrée).
- Attendez l'installation des dépendances.
- Cliquez sur l'**URL publique** (se terminant par `.gradio.live`) pour ouvrir l'interface.

---

---

## 📦 Dépendances
- `gradio` : Interface Web
- `ultralytics` : Vision YOLOv8
- `faster-whisper` : Transcription Audio
- `yt-dlp` : Téléchargement Vidéo
- `decord` : Extraction de frames ultra-rapide

---

## 📝 Licence
Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

---
Fait avec ❤️ Par Shadrak BESSANH
