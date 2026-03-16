Idées :
- Essayer de connecter des amas par voxel tree et clusteriser par amas.
- Réduire la taille de la box pour avoir une meilleure précision en densité sur HDBSCAN.

------

Pour la séparation avec HDBSCAN, on peut jouer sur deux paramètres :

<img width="894" height="620" alt="image" src="https://github.com/user-attachments/assets/b9bc9e16-b32b-4f1a-9f2a-cdff9e2d2e19" />

### `min_cluster_size` :                                                                                                                 
                                                                                                                                      
  C'est le paramètre le plus important. Il définit la taille minimale qu'un groupe de points doit avoir pour être considéré comme un cluster plutôt que du bruit.                                                                                                        
                                                                                                                                    
  HDBSCAN construit d'abord une hiérarchie complète de clusters à toutes les échelles de densité, puis "coupe" cette hiérarchie. Tout groupe qui n'a jamais atteint min_cluster_size points en même temps est rejeté comme bruit.                                         

  Exemple concret :
  - `min_cluster_size=200` → une feuille avec 150 points sera classée bruit (-1)
  - `min_cluster_size=50` → cette même feuille devient un cluster

  Sur le nuage : la densité de points varie selon la distance au scanner. Une feuille proche peut avoir 1000 pts, la même feuille loin peut en avoir 80. Un min_cluster_size trop grand va rater les feuilles distantes.

  ---
### `min_samples`

Contrôle à quel point un point doit être "entouré" pour ne pas être considéré comme bruit. Techniquement, c'est le nombre de voisins qu'un point doit avoir dans son voisinage de densité pour être un "core point".

  - `min_samples petit` (1–3) → HDBSCAN est permissif, classe peu de points en bruit, les clusters s'étendent jusqu'aux bords
  - `min_samples grand` (10–20) → HDBSCAN est conservateur, les zones de faible densité (bords de feuilles, pointes) sont rejetées en bruit

  Relation entre les deux :
  - `min_samples` par défaut = `min_cluster_size` (si non spécifié)
  - En pratique on met `min_samples` bien inférieur à min_cluster_size : par exemple `min_cluster_size=200, min_samples=5` dit "je veux des clusters d'au moins 200 pts, mais un point isolé avec juste 5 voisins proches n'est pas forcément du bruit"
