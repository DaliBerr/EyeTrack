# Rapport technique EyeTrack

## Basé sur la branche `QAT`

## 1. Introduction

EyeTrack est un projet de vision par ordinateur conçu pour estimer l’endroit où une personne regarde en analysant des images de l’œil. En termes simples, le système apprend d’abord à reconnaître les parties importantes de l’œil, mesure ensuite leur géométrie, puis convertit ces mesures en direction de regard ou en point de regard.

Ce dépôt n’est pas seulement un projet d’entraînement de réseau de neurones. Il s’agit d’une chaîne technique complète comprenant :

* la préparation du jeu de données,
* l’entraînement du réseau de neurones,
* la quantification du modèle,
* la prédiction sur des images d’œil,
* l’extraction de la géométrie de l’iris et de la pupille,
* l’analyse et le lissage de séquences,
* une démonstration temps réel sur PC,
* et une chaîne de déploiement temps réel sur Raspberry Pi.

La branche `QAT` est la branche correcte à étudier, car elle contient l’implémentation la plus récente et la mieux structurée. Par rapport à une branche plus ancienne et plus expérimentale, cette branche organise le code en modules réutilisables et introduit un flux de déploiement plus clair.

Ce rapport est rédigé pour des lecteurs non techniques. L’objectif n’est pas seulement de décrire quels fichiers existent, mais d’expliquer ce que fait le système, pourquoi chaque partie existe, et comment fonctionne la chaîne complète en temps réel.

---

## 2. Objectif principal du projet

L’objectif principal d’EyeTrack est d’estimer le regard à partir d’images de l’œil.

Le système fonctionne en trois grandes étapes :

1. **Segmentation de l’œil**
   Un réseau de neurones analyse une image de l’œil et classe chaque pixel dans plusieurs catégories, par exemple le fond, la frontière externe de l’œil, l’iris et la pupille.

2. **Extraction de caractéristiques géométriques**
   Après la segmentation, le logiciel mesure les formes et les positions, en particulier le centre et l’ellipse de l’iris et de la pupille.

3. **Estimation du regard et projection**
   Les caractéristiques oculaires mesurées sont transformées en caractéristique de direction du regard, puis, après une étape de calibration, cette caractéristique est projetée en un point sur l’écran.

Le projet prend en charge à la fois une utilisation hors ligne et une utilisation en temps réel :

* le **mode hors ligne** est utilisé pour l’entraînement, l’évaluation, la quantification et l’analyse de séquences ;
* le **mode temps réel** est utilisé pour traiter en direct des flux caméra sur PC ou sur Raspberry Pi.

---

## 3. Vue d’ensemble du système

À haut niveau, le dépôt peut être compris comme quatre sous-systèmes connectés.

### 3.1 Sous-système d’entraînement

Cette partie entraîne un modèle U-Net léger sur un jeu de données de type OpenEDS.

### 3.2 Sous-système géométrique

Cette partie convertit les cartes de segmentation en mesures oculaires exploitables, telles que le centre de la pupille, le centre de l’iris, les axes des ellipses, les décalages normalisés et les rapports d’aire.

### 3.3 Sous-système de déploiement

Cette partie exporte les modèles vers ONNX, les quantifie, évalue leur qualité et mesure leur latence d’exécution.

### 3.4 Sous-système temps réel

Cette partie exécute le modèle en direct sur des images caméra. Elle existe sous deux formes :

* une **démonstration temps réel sur PC** ;
* et une **chaîne temps réel à deux caméras sur Raspberry Pi**.

La version Raspberry Pi constitue le chemin de déploiement le plus important de ce dépôt.

---

## 4. Outils, frameworks et bibliothèques principaux

Ce projet repose sur une combinaison de bibliothèques de machine learning, de traitement d’image, d’analyse de données et d’outils multimédias côté embarqué.

### 4.1 Noyau Python et apprentissage profond

#### PyTorch

PyTorch est utilisé pour définir, entraîner et affiner le réseau de neurones. Il fournit :

* le calcul tensoriel,
* la définition du modèle,
* l’entraînement par rétropropagation,
* la sauvegarde et le chargement de checkpoints,
* et la préparation à l’entraînement conscient de la quantification.

#### ONNX

ONNX est utilisé comme format neutre de modèle pour le déploiement. Un modèle entraîné sous PyTorch peut être exporté en ONNX afin d’être utilisé dans un environnement d’exécution plus portable.

#### ONNX Runtime

ONNX Runtime est utilisé pour l’inférence de déploiement et pour la quantification. Il est particulièrement important ici, car le déploiement Raspberry Pi n’exécute pas directement le modèle PyTorch. À la place, il exécute un modèle ONNX quantifié avec ONNX Runtime.

### 4.2 Vision par ordinateur et traitement d’image

#### OpenCV

OpenCV est utilisé pour :

* le redimensionnement d’images,
* la conversion en niveaux de gris,
* l’extraction de contours,
* l’ajustement d’ellipses,
* les opérations morphologiques,
* la visualisation,
* et l’accès caméra sur PC.

Il est également utilisé pour dessiner les superpositions en direct : ellipses, cibles de calibration, flèches et panneaux d’aperçu écran.

#### Pillow

Pillow est utilisé principalement pour lire les fichiers image dans les scripts de jeu de données et d’inférence.

### 4.3 Analyse de données et visualisation

#### NumPy

NumPy constitue la base numérique du projet. Il est utilisé pour les tableaux d’images, les calculs géométriques, l’interpolation, le lissage de signaux et les calculs de calibration.

#### Pandas

Pandas est utilisé pour le traitement des fichiers CSV de séquences et les workflows de filtrage.

#### Matplotlib

Matplotlib est utilisé pour générer des graphiques d’analyse comparant les signaux bruts et nettoyés dans le temps.

### 4.4 Outils utilitaires

#### tqdm

Utilisé pour les barres de progression pendant l’entraînement et la validation.

#### argparse

Utilisé dans presque tous les scripts pour définir l’interface en ligne de commande.

### 4.5 Outils spécifiques Raspberry Pi

#### Picamera2

Picamera2 est utilisé sur Raspberry Pi pour accéder aux caméras CSI de manière moderne, synchronisée et avec prise en compte des métadonnées.

#### libcamera

Utilisé avec Picamera2 pour la configuration des caméras et les métadonnées des images.

#### GStreamer

GStreamer est utilisé sur Raspberry Pi pour deux sorties :

* le streaming vidéo RTSP,
* l’enregistrement vidéo local en H.264.

#### python3-gi / GstRtspServer

Utilisés pour construire le serveur RTSP sur Raspberry Pi.

### 4.6 Pourquoi cette combinaison d’outils est importante

Il s’agit d’un choix de conception important :

* **PyTorch** est idéal pour l’entraînement ;
* **ONNX Runtime** est plus adapté au déploiement léger ;
* **OpenCV** gère le traitement et la visualisation temps réel ;
* **Picamera2 + GStreamer** rendent la chaîne Raspberry Pi praticable en conditions réelles.

Ainsi, le dépôt n’est pas seulement un dépôt de modèle. C’est un pont entre le développement en machine learning et le déploiement embarqué réel.

---

## 5. Organisation du dépôt

Le dépôt est structuré autour d’un package Python central nommé `eyetrack`, tandis que de nombreux scripts à la racine servent de points d’entrée en ligne de commande.

### 5.1 Scripts d’entrée principaux

Les scripts principaux sont notamment :

* `Train.py` — point d’entrée d’entraînement,
* `predict_segmentation_to_npy.py` — prédiction sur images,
* `extract_geometry_from_segmentation.py` — extraction géométrique,
* `batch_predict_and_extract_sequences.py` — traitement par lots des séquences,
* `baseline_filter_sequences_v3.py` — nettoyage et lissage de séquences,
* `analyze_sequence_geometry.py` — visualisation et analyse d’anomalies,
* `export_unet_to_onnx.py` — export ONNX,
* `quantize_onnx_model.py` — quantification statique ONNX,
* `evaluate_onnx_model.py` — évaluation ONNX,
* `benchmark_onnx_model.py` — mesure de latence,
* `realtime_eye_direction.py` — démonstration temps réel sur PC,
* `realtime_eye_direction_pi.py` — système temps réel Raspberry Pi.

### 5.2 Structure interne du package

Le package `eyetrack` constitue le vrai cœur du projet. Il contient :

* `config.py` — paramètres globaux et helpers de métadonnées du modèle,
* `data/` — chargement du jeu de données et prétraitement,
* `models/` — le modèle U-Net,
* `training/` — checkpoints, moteur d’entraînement, utilitaires QAT,
* `metrics/` — métriques de segmentation,
* `deployment/` — helpers liés à ONNX,
* `gaze.py` — extraction géométrique et mathématiques de calibration,
* `realtime_gaze.py` — helpers de calibration et de dessin,
* `raspi/` — modules runtime Raspberry Pi.

Cette organisation est beaucoup plus claire qu’un prototype monolithique. Elle montre que la branche `QAT` a été pensée comme un système maintenable.

---

## 6. Modèle et pipeline d’apprentissage

### 6.1 Architecture du modèle

Le modèle utilisé est un **U-Net** léger.

U-Net est une architecture populaire pour la segmentation sémantique, car elle combine :

* un **chemin de sous-échantillonnage** qui capture le contexte,
* et un **chemin de sur-échantillonnage** qui restaure le détail spatial.

Dans cette implémentation, le modèle est volontairement léger. Sa largeur peut être ajustée via le paramètre `base_channels`, ce qui permet des modèles plus petits comme `b8` ou `b6` pour le mobile ou l’embarqué.

### 6.2 Entrée et sortie

Le modèle attend une image d’œil en niveaux de gris et produit des logits par pixel pour chaque classe.

La convention de classes est :

* classe 0 : fond,
* classe 1 : frontière externe ou région externe de l’œil,
* classe 2 : iris,
* classe 3 : pupille.

### 6.3 Format du jeu de données

L’entraînement utilise une structure de type OpenEDS :

* `train/images`, `train/labels`, `train/masks`
* `validation/images`, `validation/labels`, `validation/masks`

C’est un point important, car de nombreux scripts supposent cette organisation.

### 6.4 Processus d’entraînement

Pendant l’entraînement, chaque image est :

1. chargée en niveaux de gris,
2. normalisée,
3. redimensionnée à la taille d’entrée du modèle,
4. associée à sa carte de labels,
5. éventuellement associée à un masque de validité.

La boucle d’entraînement calcule :

* la perte cross-entropy,
* l’exactitude pixel,
* l’exactitude pixel masquée,
* le score Dice pour les classes d’avant-plan.

### 6.5 Checkpoints et métadonnées

Un point fort de la branche `QAT` est que les checkpoints ne stockent pas seulement les poids du réseau. Ils stockent aussi des métadonnées comme :

* le nombre de canaux d’entrée,
* le nombre de classes de sortie,
* la largeur de base du réseau,
* la largeur et la hauteur d’entrée,
* le mode de prétraitement,
* le réglage AMP,
* l’usage ou non des masques,
* le mode de quantification,
* le backend QAT.

C’est important, car les scripts ultérieurs peuvent retrouver automatiquement la bonne configuration du modèle à partir du checkpoint lui-même, au lieu d’exiger que l’utilisateur ressaisisse tous les paramètres.

---

## 7. Quantification et stratégie de déploiement

La quantification est un thème central de cette branche.

### 7.1 Pourquoi la quantification est nécessaire

Pour un déploiement embarqué, en particulier sur Raspberry Pi, un modèle en virgule flottante peut être trop lourd. La quantification réduit la taille du modèle et peut améliorer la vitesse pratique d’inférence sur CPU.

### 7.2 PTQ et QAT dans ce dépôt

Le projet prend en charge deux approches liées :

* **PTQ (Post-Training Quantization)** — quantifier un modèle déjà entraîné ;
* **QAT (Quantization-Aware Training)** — poursuivre l’entraînement tout en simulant les effets de la quantification.

### 7.3 Remarque pratique importante

Même si la branche s’appelle `QAT`, le dépôt ne considère pas l’export direct d’un graphe eager QAT vers ONNX QDQ comme le chemin principal et stable de déploiement.

À la place, la chaîne recommandée est :

1. entraîner ou affiner un modèle PyTorch,
2. éventuellement exécuter un fine-tuning QAT,
3. retirer la préparation QAT pour revenir à un modèle flottant fusionné,
4. exporter un modèle ONNX flottant,
5. appliquer une quantification statique via ONNX Runtime,
6. déployer le modèle ONNX quantifié résultant.

Il s’agit d’un choix d’ingénierie très important. Il montre que les auteurs privilégient la fiabilité du déploiement plutôt qu’une solution théoriquement plus élégante mais moins stable.

---

## 8. Extraction géométrique hors ligne

Après la segmentation, le projet ne déduit pas directement un point de regard. Il convertit d’abord la carte de segmentation en information géométrique.

### 8.1 Ce qui est extrait

À partir de la carte de labels prédite, le code extrait :

* le masque de l’iris,
* le masque de la pupille,
* le masque de frontière externe,
* les composantes connexes,
* les masques nettoyés,
* les centroïdes,
* les ellipses ajustées,
* les aires,
* les décalages normalisés entre pupille et iris,
* le rapport d’aire pupille/iris.

### 8.2 Pourquoi c’est important

Cette représentation géométrique intermédiaire fait le lien entre la segmentation et l’estimation du regard.

Au lieu d’utiliser directement les pixels bruts pour la projection du regard, le projet utilise des caractéristiques interprétables comme :

* la position du centre de la pupille par rapport au centre de l’iris,
* la distance de la pupille par rapport au centre de l’œil,
* et la stabilité de la géométrie.

Cela rend le système plus explicable, plus facile à déboguer et plus stable à calibrer.

---

## 9. Traitement et nettoyage des séquences

Le dépôt prend également en charge le traitement par lots de séquences d’images oculaires stockées dans des dossiers comme `S_0`, `S_1`, etc.

### 9.1 Workflow de séquences

Pour chaque séquence, le système peut :

1. exécuter la segmentation sur toutes les images,
2. sauvegarder les cartes de labels `.npy`,
3. extraire la géométrie dans `geometry.csv`,
4. détecter les images anormales,
5. interpoler les courtes séquences dégradées,
6. préserver les longues coupures comme valeurs manquantes,
7. lisser les segments valides,
8. générer des graphiques et des fichiers CSV de synthèse.

### 9.2 Pourquoi le nettoyage des séquences est nécessaire

Le suivi du regard réel est bruité. Certaines images peuvent être corrompues par :

* l’occlusion par la paupière,
* le flou,
* des erreurs de segmentation,
* une pupille instable,
* ou des sauts soudains dans les caractéristiques extraites.

Le filtrage des séquences n’est donc pas accessoire. C’est une étape importante de stabilisation.

---

## 10. Chaîne temps réel sur PC

Le système temps réel côté PC est implémenté dans `realtime_eye_direction.py`.

Ce script sert principalement d’environnement de démonstration et de débogage.

### 10.1 Rôle du pipeline PC

La version desktop est utile pour :

* tester un modèle entraîné avec une webcam,
* sélectionner manuellement une région de l’œil,
* valider la qualité de la segmentation,
* tester le comportement de la calibration,
* visualiser la géométrie extraite,
* et comparer un chemin d’exécution `.pth` ou `.onnx`.

Il s’agit donc à la fois d’une démonstration et d’un outil de validation pour le développeur.

### 10.2 Source d’entrée

Le script ouvre une caméra via OpenCV et capture des images complètes.

Comme l’image entière de la webcam contient bien plus que l’œil, l’utilisateur doit d’abord sélectionner une **région d’intérêt (ROI)** à la souris. Cette ROI devient l’image de travail pour toutes les étapes suivantes.

### 10.3 Chaîne temps réel PC étape par étape

Le pipeline PC peut être décrit de la manière suivante.

#### Étape 1 : capture caméra

Une image webcam est capturée via OpenCV.

#### Étape 2 : sélection de la ROI

L’utilisateur dessine un rectangle autour de l’œil. Cette région sélectionnée devient l’entrée de travail.

#### Étape 3 : prétraitement

La ROI est convertie en niveaux de gris, redimensionnée à la taille d’entrée du modèle, puis transformée en tenseur.

#### Étape 4 : inférence de segmentation

Le système exécute soit :

* un checkpoint PyTorch,
* soit un modèle ONNX.

Il produit alors une carte de segmentation.

#### Étape 5 : extraction géométrique

La carte de labels est convertie en géométrie de l’iris et de la pupille grâce au code partagé d’extraction géométrique.

#### Étape 6 : sélection de caractéristique

Le système dérive une caractéristique liée au regard à partir de la géométrie. Il prend en charge deux modes :

* `pupil_iris` — basé sur le décalage normalisé pupille/iris,
* `iris_only` — basé sur la position de l’iris et les informations de frontière.

#### Étape 7 : estimation de confiance

Un suivi de qualité estime à quel point l’image courante est fiable.

Cette confiance dépend de facteurs comme :

* la stabilité de l’aire de l’iris,
* la stabilité du mouvement,
* la qualité de la forme,
* et la validité des caractéristiques requises.

#### Étape 8 : lissage temporel

Un filtre de Kalman adaptatif lisse la caractéristique dans le temps. Cela réduit les tremblements.

#### Étape 9 : calibration

L’utilisateur peut lancer une procédure de calibration à neuf points. Pendant cette calibration, le système affiche des cibles et enregistre les caractéristiques oculaires associées à des points connus de l’écran.

#### Étape 10 : prédiction du point écran

Après la calibration, la caractéristique lissée est projetée en une coordonnée écran normalisée.

#### Étape 11 : visualisation

Le script dessine :

* les ellipses de l’iris et de la pupille,
* les centres,
* des flèches de direction,
* un petit panneau d’aperçu écran,
* un aperçu de la segmentation,
* un aperçu de la ROI,
* du texte de débogage.

### 10.4 Interprétation du pipeline PC

Le script temps réel desktop doit être compris comme un laboratoire interactif pour l’eye tracker. Il permet de voir tous les résultats intermédiaires importants, ce qui est extrêmement utile pour le débogage.

---

## 11. Chaîne temps réel sur Raspberry Pi

Le système Raspberry Pi est la partie temps réel la plus importante du dépôt.

Il est implémenté principalement dans :

* `realtime_eye_direction_pi.py`,
* et le package `eyetrack/raspi/`.

Contrairement à la démonstration desktop, il s’agit ici d’une architecture orientée déploiement.

### 11.1 Idée générale

La version Raspberry Pi est construite autour de **deux flux caméra différents** :

1. une **caméra œil** pour le suivi rapproché en infrarouge,
2. une **caméra FPV** pour la scène regardée vers l’avant.

Le projet n’affiche pas simplement une flèche de regard sur l’image de l’œil. Il utilise la caméra œil pour estimer le regard, puis publie ce regard comme métadonnée pouvant être associée au flux FPV.

Cela signifie que le système Raspberry Pi combine :

* l’acquisition,
* l’inférence,
* la synchronisation,
* la calibration,
* la sortie de métadonnées,
* et le streaming ou l’enregistrement vidéo.

### 11.2 Pourquoi le pipeline Pi est séparé du pipeline PC

Le script Raspberry Pi ne réutilise pas directement le script desktop, car les contraintes de déploiement sont différentes.

Le côté Pi doit gérer :

* des caméras CSI plutôt que des webcams génériques,
* un déploiement ONNX fixe plutôt qu’un test flexible PyTorch,
* une inférence CPU à faible latence,
* des timestamps synchronisés sur deux caméras,
* un serveur RTSP optionnel,
* un enregistrement local optionnel,
* et une sortie de métadonnées lisibles par machine.

Il ne s’agit donc pas d’un simple portage. C’est une architecture de système distincte.

---

## 12. Modules du pipeline Raspberry Pi

L’implémentation Raspberry Pi est découpée en plusieurs modules, chacun responsable d’une partie précise du runtime.

### 12.1 `runtime.py`

Ce module charge le modèle ONNX quantifié et exécute l’inférence.

Ses responsabilités sont :

* vérifier que le modèle est bien un fichier ONNX,
* vérifier qu’il s’agit bien d’un modèle quantifié de type QDQ,
* créer une session ONNX Runtime,
* lire les métadonnées du modèle,
* imposer une entrée en niveaux de gris à un seul canal,
* exécuter l’inférence de segmentation et produire des cartes de labels.

C’est important, car cela définit clairement le runtime Pi comme un chemin de déploiement ONNX uniquement.

### 12.2 `camera.py`

Ce module gère l’acquisition via Picamera2.

Ses responsabilités sont :

* détecter les caméras disponibles,
* configurer chaque flux,
* capturer les images dans des threads dédiés,
* conserver les timestamps issus des métadonnées,
* convertir les images YUV420 de l’œil en tenseurs d’entrée modèle,
* convertir les images FPV en BGR si nécessaire.

La caméra œil utilise YUV420 et extrait directement le plan de luminance, ce qui est un choix intelligent, car le modèle n’a besoin que d’une image en niveaux de gris.

### 12.3 `sync.py`

Ce module gère la synchronisation et la fraîcheur des images.

Ses responsabilités sont :

* résoudre les timestamps à partir des métadonnées,
* associer les images œil et FPV,
* calculer le décalage temporel entre les deux flux,
* détecter les images obsolètes.

C’est essentiel, car dans un système à deux caméras, une estimation du regard n’a de sens que si elle est associée à la bonne image de scène.

### 12.4 `metadata.py`

Ce module construit et publie les métadonnées de regard.

Le paquet de métadonnées contient :

* les timestamps œil et FPV,
* les coordonnées écran prédites,
* l’état de calibration,
* la validité du suivi,
* la validité de la caractéristique,
* la cadence FPS,
* la latence d’inférence,
* l’état de synchronisation,
* l’état d’obsolescence des flux,
* et un message d’état.

Les métadonnées peuvent être publiées :

* vers la sortie standard en JSON ligne par ligne,
* ou via UDP.

Cette conception est particulièrement solide, car elle sépare **le transport vidéo** du **transport de l’information de regard**.

### 12.5 `rtsp.py`

Ce module crée un serveur RTSP basé sur GStreamer.

Son rôle est de prendre les images FPV et de les diffuser sur le réseau sous forme de vidéo H.264.

Il privilégie les encodeurs matériels et peut basculer vers un encodage logiciel si nécessaire.

### 12.6 `recorder.py`

Ce module enregistre localement la vidéo FPV dans des fichiers MKV.

Point important : la version enregistrée n’est pas identique au flux RTSP :

* en mode RTSP, la vidéo FPV est transmise sans incrustation permanente du regard,
* en mode enregistrement, une marque légère de regard est dessinée dans la vidéo locale.

C’est un choix très pertinent. Il permet une visualisation distante en direct sans altérer définitivement la vidéo diffusée, tout en proposant des enregistrements annotés pour une revue ultérieure.

### 12.7 `overlay.py`

Ce module dessine des éléments visuels sur les images FPV.

Il peut dessiner :

* un point de regard,
* un cercle de regard pour l’enregistrement,
* une cible de calibration,
* et un panneau de débogage.

Différentes fonctions d’overlay sont utilisées selon le mode de sortie.

### 12.8 `terminal.py`

Ce module fournit une lecture non bloquante du clavier, afin que l’utilisateur puisse envoyer des commandes pendant l’exécution du pipeline.

---

## 13. Chaîne temps réel Raspberry Pi étape par étape

C’est la section la plus importante du rapport.

La chaîne temps réel Raspberry Pi peut être décrite ainsi.

### Étape 1 : chargement du modèle

Le runtime Pi charge un **modèle ONNX quantifié** et vérifie que sa forme d’entrée correspond exactement à la résolution configurée de la caméra œil.

### Étape 2 : détection et démarrage des caméras

Le script détecte les caméras Picamera2 disponibles et démarre :

* un flux infrarouge proche de l’œil,
* et éventuellement un flux FPV.

Chaque flux fonctionne dans son propre thread.

### Étape 3 : acquisition des images œil

La caméra œil capture en continu des images au format YUV420. Le logiciel en extrait le plan de luminance comme image de l’œil en niveaux de gris.

Cette approche est efficace, car elle évite des conversions couleur inutiles.

### Étape 4 : acquisition des images FPV

La caméra FPV capture séparément les images de la scène observée.

### Étape 5 : association temporelle et vérification d’état

Le système lit les timestamps issus des métadonnées des caméras et compare les temps des images œil et FPV. Il vérifie aussi si l’un des deux flux est devenu obsolète.

Cela évite d’associer une estimation de regard à une image de scène ancienne ou incorrecte.

### Étape 6 : inférence de segmentation

Chaque nouvelle image œil est normalisée puis envoyée au modèle ONNX quantifié. Le modèle retourne une carte de segmentation.

### Étape 7 : extraction géométrique

La carte de labels est convertie en géométrie d’iris et de pupille. Cette étape est partagée avec le pipeline hors ligne et le pipeline PC, ce qui est une bonne décision de conception, car la logique géométrique reste cohérente entre les environnements.

### Étape 8 : extraction de caractéristique et confiance

Une caractéristique de suivi est sélectionnée à partir de la géométrie, en mode `pupil_iris` ou `iris_only`. Le système calcule ensuite un score de confiance pour estimer si cette caractéristique est exploitable.

### Étape 9 : filtrage temporel

Un filtre de Kalman lisse la caractéristique dans le temps afin de réduire l’instabilité et les tremblements.

### Étape 10 : calibration

Lorsque l’utilisateur lance la calibration, le système démarre une session à neuf points.

Pour chaque cible :

* l’utilisateur fixe un point connu,
* le système attend un temps de stabilisation,
* des échantillons valides sont collectés,
* les échantillons sont filtrés et moyennés,
* puis l’ensemble des points collectés sert à ajuster un modèle de projection.

Le projet peut ajuster un modèle affine ou un modèle polynomial selon la quantité d’information de calibration disponible.

### Étape 11 : prédiction du regard

Une fois la calibration terminée, la caractéristique lissée est convertie en position écran normalisée `(u, v)`.

### Étape 12 : création des métadonnées

Le résultat courant du regard, les informations d’état, l’état de synchronisation et les informations temporelles sont regroupés dans un paquet de métadonnées.

### Étape 13 : choix du mode de sortie

Le système prend en charge deux modes de sortie.

#### Mode RTSP

En mode RTSP :

* la vidéo FPV est diffusée,
* le regard n’est pas incrusté de manière permanente dans l’image diffusée,
* les métadonnées sont émises séparément.

Ce mode est utile pour des systèmes distants qui souhaitent superposer le regard à l’extérieur.

#### Mode enregistrement

En mode enregistrement :

* le flux FPV est enregistré localement,
* un cercle léger représentant le regard est dessiné dans la vidéo enregistrée,
* l’enregistrement n’est autorisé qu’après la fin de la calibration.

### Étape 14 : aperçu local optionnel de l’œil

Un aperçu local de l’œil peut être activé. Il affiche :

* l’image de l’œil,
* la géométrie ajustée,
* l’état de calibration,
* un aperçu écran,
* la confiance,
* et l’état d’exécution.

Il s’agit d’un outil de débogage, et non du canal de sortie principal.

---

## 14. Pourquoi la conception Raspberry Pi est solide

Le pipeline Raspberry Pi présente plusieurs bonnes idées d’ingénierie.

### 14.1 Séparation claire des responsabilités

Le système sépare :

* la capture caméra,
* l’inférence,
* la synchronisation,
* la calibration,
* le transport vidéo,
* et le transport des métadonnées.

Cela rend le code plus facile à maintenir et à faire évoluer.

### 14.2 Chemin d’entrée efficace en niveaux de gris

La caméra œil utilise directement le plan Y issu du YUV420. Cela évite un prétraitement inutile.

### 14.3 Exigence d’un modèle quantifié

Le runtime Pi n’accepte que des modèles ONNX quantifiés. C’est une bonne règle de déploiement, car elle empêche de lancer par erreur un modèle de développement trop lourd sur le dispositif.

### 14.4 Double stratégie de sortie

Le système distingue :

* le **streaming**, où le regard reste une métadonnée externe,
* et l’**enregistrement**, où un petit marqueur de regard est intégré.

C’est une conception très pertinente pour un usage réel.

### 14.5 Prise en compte explicite de la synchronisation

Le code suit explicitement le décalage temporel et les états obsolètes. C’est très important dans tout système temps réel multi-caméras.

### 14.6 Calibration intégrée au déploiement

La calibration n’est pas traitée comme un simple outil de recherche hors ligne. Elle est intégrée directement au runtime temps réel, ce qui rend le système plus pratique.

---

## 15. Comparaison entre les pipelines temps réel PC et Raspberry Pi

Même si les deux pipelines réalisent un suivi du regard en temps réel, leur rôle est différent.

### Pipeline PC

La version PC sert surtout à :

* l’expérimentation,
* la validation,
* le débogage,
* l’inspection visuelle,
* et les tests flexibles avec des modèles `.pth` ou `.onnx`.

Elle est plus interactive et davantage orientée développeur.

### Pipeline Raspberry Pi

La version Raspberry Pi sert surtout à :

* le déploiement,
* le fonctionnement temps réel avec deux caméras,
* la génération de métadonnées en direct,
* le streaming RTSP,
* l’enregistrement local,
* et l’exécution embarquée.

Elle est plus orientée système et exploitation.

En résumé :

* la **version PC** est un banc d’essai temps réel,
* la **version Pi** est un prototype de système embarqué temps réel.

---

## 16. Scripts hors ligne à mentionner malgré tout

Même si ce rapport met l’accent sur les pipelines temps réel, plusieurs scripts hors ligne restent importants car le système temps réel dépend d’eux.

### 16.1 `Train.py`

Utilisé pour entraîner le modèle de segmentation.

### 16.2 `predict_segmentation_to_npy.py`

Utilisé pour générer des cartes de labels prédites sur des dossiers d’images.

### 16.3 `extract_geometry_from_segmentation.py`

Utilisé pour convertir les résultats de segmentation en fichiers CSV géométriques.

### 16.4 `batch_predict_and_extract_sequences.py`

Utilisé pour traiter automatiquement des séquences entières d’images.

### 16.5 `baseline_filter_sequences_v3.py`

Utilisé pour nettoyer les signaux géométriques bruités en détectant les images aberrantes, en interpolant les courtes séquences dégradées, en conservant les longues coupures et en lissant les régions valides.

### 16.6 `analyze_sequence_geometry.py`

Utilisé pour générer des graphiques d’analyse et des rapports de séquences.

### 16.7 `export_unet_to_onnx.py`

Utilisé pour convertir un modèle PyTorch entraîné vers ONNX.

### 16.8 `quantize_onnx_model.py`

Utilisé pour réaliser la quantification statique ONNX, nécessaire au déploiement Raspberry Pi.

### 16.9 `evaluate_onnx_model.py`

Utilisé pour évaluer la qualité des modèles ONNX quantifiés et éventuellement les comparer au modèle PyTorch.

### 16.10 `benchmark_onnx_model.py`

Utilisé pour mesurer la latence d’inférence sous ONNX Runtime.

Ces outils soutiennent le système temps réel même s’ils ne font pas partie de la boucle d’exécution directe.

---

## 17. Limites et remarques pratiques importantes

Plusieurs limites pratiques importantes doivent être explicitées.

### 17.1 Le système dépend de la qualité de la segmentation

Si le modèle segmente mal l’iris et la pupille, toute la chaîne d’estimation du regard devient instable.

### 17.2 L’estimation temps réel dépend de la calibration

Sans calibration, le système peut calculer des caractéristiques internes de l’œil, mais il ne peut pas les projeter de manière fiable en un point écran.

### 17.3 Le runtime Raspberry Pi exige un modèle ONNX quantifié

Ce n’est pas optionnel dans la conception actuelle.

### 17.4 Le PC et le Pi ont des rôles différents

Le script desktop n’est pas un remplaçant direct du déploiement Pi. Le runtime Pi est plus spécialisé et plus contraint.

### 17.5 Le nettoyage des séquences reste important

Même avec un bon modèle, les données réelles de l’œil contiennent des valeurs aberrantes et des images instables. C’est pourquoi le dépôt inclut des outils dédiés au filtrage temporel.

---

## 18. Conclusion

La branche `QAT` d’EyeTrack est un projet de suivi du regard bien structuré, qui combine apprentissage profond, extraction géométrique, quantification et déploiement temps réel.

Sa contribution la plus importante n’est pas seulement le modèle de segmentation lui-même, mais le système complet construit autour de ce modèle.

Le dépôt prend en charge toute une chaîne :

* l’entraînement d’un modèle léger de segmentation de l’œil,
* sa préparation pour un déploiement efficace,
* l’extraction d’une géométrie oculaire interprétable,
* le lissage et l’analyse de séquences temporelles,
* l’exécution d’un suivi en direct sur PC pour le débogage,
* et le déploiement d’un système temps réel à deux caméras sur Raspberry Pi.

Le pipeline Raspberry Pi est la partie la plus forte et la plus orientée déploiement du projet. Il a été conçu comme un runtime embarqué modulaire combinant :

* un modèle oculaire ONNX quantifié,
* une capture double caméra CSI,
* une logique de synchronisation,
* un suivi de caractéristiques pondéré par la confiance,
* une calibration à neuf points,
* une publication de métadonnées,
* un streaming RTSP,
* et un enregistrement local annoté.

Pour un lecteur non technique, la manière la plus simple de comprendre EyeTrack est la suivante :

**le projet observe l’œil, identifie l’iris et la pupille, mesure leur mouvement, apprend comment ces mouvements correspondent à une direction de regard, puis injecte ce résultat dans un système vidéo temps réel.**

Ainsi, EyeTrack n’est pas seulement une expérience de réseau de neurones, mais un prototype complet de système de vision sensible au regard en temps réel.

---

## 19. Ordre de lecture conseillé pour les nouveaux venus

Pour une personne découvrant le dépôt, l’ordre de lecture le plus utile est :

1. `README.md`
2. `realtime_eye_direction_pi.py`
3. `eyetrack/raspi/runtime.py`
4. `eyetrack/raspi/camera.py`
5. `eyetrack/raspi/metadata.py`
6. `eyetrack/gaze.py`
7. `realtime_eye_direction.py`
8. `eyetrack/models/unet.py`
9. `eyetrack/workflows/train.py`
10. les scripts ONNX et de quantification

Cet ordre permet de comprendre d’abord le système final déployé, puis de remonter vers le modèle et les détails d’entraînement.

---

## 20. Résumé final en un paragraphe

EyeTrack est un système modulaire de suivi du regard construit autour d’un modèle léger de segmentation U-Net. Il détecte les régions de l’iris et de la pupille, extrait des caractéristiques géométriques, les calibre en coordonnées écran, puis délivre une information de regard en temps réel. Le pipeline desktop sert surtout d’environnement de test et de visualisation, tandis que le pipeline Raspberry Pi constitue un système embarqué dédié utilisant un modèle ONNX quantifié, deux caméras CSI, une logique de synchronisation, une sortie de métadonnées, un streaming RTSP et un enregistrement local optionnel. La branche `QAT` représente donc une chaîne d’ingénierie complète allant du développement du modèle jusqu’au suivi du regard déployable en temps réel.
