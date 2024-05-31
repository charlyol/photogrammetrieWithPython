import os
import shutil

# Définir le chemin du dossier source et du dossier de destination
source_dir = '../data/frames'
dest_dir = '../data/frames_shutil'

# Obtenir la liste de tous les fichiers dans le dossier source
files = os.listdir(source_dir)

# Sélectionner une image sur 50
selected_files = files[::50]

# Déplacer les fichiers sélectionnés vers le dossier de destination
for file in selected_files:
    shutil.move(os.path.join(source_dir, file), dest_dir)
