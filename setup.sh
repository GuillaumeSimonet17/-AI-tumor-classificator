#!/bin/bash

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Environnement virtuel déjà présent. Activation..."
else
    echo "Création de l'environnement virtuel..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

if [ -f "requirements.txt" ]; then
    echo "Installation des dépendances depuis requirements.txt..."
    pip install -r requirements.txt
else
    echo "Aucun fichier requirements.txt trouvé."
fi

echo "L'environnement virtuel est activé et les dépendances sont installées."
