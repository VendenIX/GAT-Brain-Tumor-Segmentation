# **Multi-class Brain Tumor Segmentation using Graph Attention Network**
(Graph Attention Network pour la segmentation de glioblastomes)

## What is GAT ? 
Il y a deux familles de Graph Neural Networks : 
- Méthodes spectral (plus mathématique et excessif computationnellement (continue), avec la famille la plus connue : les GCN Graph Convolutionnals Networks. Cela repose sur la théorie spectrale des graphes en utilisant des propriétés mathématiques de la décomposition spectrale de la matrice de Laplace associée au graphe. On peut notamment utiliser la transformée de Fourier (utilisation des vecteurs propres de la matrice de Laplace du graphe). Sauf que c'est très couteux donc tout le monde fait des approximations un peu complexes (Polynomes de Chebyshev, etc)
- Méthodes spatiales (nous ici (GAT fait partie de la famille des méthodes spatiales) ces méthodes effectuent les opérations directement dans l'espace en aggrégent les informations des voisins **locaux**. En bref on considère le voisinage des noeuds. Je vais présenter GAT plus loin.

Ce notebook est une **tentative** d'implémentation du GAT issu de l'article 10.48550/arXiv.2302.05598 par Dhrumil Patel et al.
J'ai beaucoup été freiné par les capacités de ma machine pour faire tourner tout ça, je n'ai pas pu faire d'entraînement, mais j'ai au moins pu chargé les données dans une structure d'arbre entraînables par le modèle à quelques ajustements prêts. Ce qu'il faut surtout consolider dans cette tentative c'est la séparation 3D en supervoxels qui est délicate et un problème assez complexe.

#### Présentation du dataset : 
Le dataset est le dataset très connu d'IRM multimodalitées BRATS2021 qui possède des tumeurs annotées de types glioblastomes qui sont des tumeurs invasives très aggressives.
Les différentes modalitées sont les suivantes : 
T1 : séquence de base pour l'anatomie cérébrale
flair : une T2 avec le liquide ancéphalorachien atténué (le truc noir au milieu du cerveau), ce qui facilite la détection des lésions près des ventricules
T1ce : une T1 avec agent de contraste (le mieux pour détecter les glioblastomes car cela fait ressortir les zones présentant une barrière hémato-encéphalique altérée (les tumeurs))
T2 : plus claire, permet de récupérer les oedèmes ou des lésions hyperintenses 

#### Data curation : 
J'ai utilisé toutes les IRM pour avoir plusieurs features dans chaque noeud de mes graphes. Je voulais utiliser les transformations de la librairie MONAI pour normaliser mes données et autres ajustements.

#### Segmentation en supervoxels (SLIC) 

##### SLIC
Comment optimiser trouver le bon compromis pour espérer lancer un entraînement sur une petite machine ?

Les moyens que j'ai sont les suivants : 
- Passer de la 3D à de la 2D, ou connecter le graphe principalement de tout en bas sur le volume mais moins entre les slices elles-mêmes
- optimiser SLIC en 3d au lieu de boucler slice par slice 
- optimiser la construction du graphe au lieu de faire une triple boucle sur x y z 

Réduire le nombre de supervoxels n'est pas forcément une bonne idée car on perd vite en précision 

Une autre idée est de faire un sous-échantillonnage spatial -> c'est à dire diviser par 2 la résolution des images par exemple
L'article original combine les quatre modalités IRM en un volume 3D global. 
J'ai utilisé l'algorithme très connu SLIC que j'ai utilisé notamment pour mon projet de recherche en immersion pour la technique d'explicabilité LIME. Cet algorithme est particulièrement efficace pour donner une partition de l'image en zones ressemblantes pour grouper les 
pixels et ainsi diminuer le nombre de noeuds. Car au final, chaque IRM sera considérée comme un graphe, et chaque ensemble de pixels appelées supervoxels seront les noeuds aux caractéristiques diverses (20 caractéristiques que nous verrons juste après). 
Dans l'article ils utilisent SLIC en 3-D et ils disent avoir utilisé des techniques de clustering comme K-Mean pour obtenir k supervoxels. 
Mon implémentation s'en approche sauf que j'ai moyenné donc je perds un peu d'informations inter-coupes je crois mais c'était plus simple de faire ainsi. 

L'article à fait une grid_search pour optimiser les hyperparamètres alpha, beta, etc pour la distance, et un paramètre lambda. 
Pour me simplifier la tâche j'ai uniquement utilisé compactness et n_segments. 

##### Extraction des features 
L'article a choisi d'extraire des percentiles des intensités (10, 25, 50, 75, 90) par canal (toutes les IRM différentes). Ca fait donc 5 * 4 canal = 20 caractéristiques par noeud dans chaque graphe (il y a des milliers de noeuds). J'ai fait pareil, et j'ai regroupé tout ça en un seul vecteur 1d de caractéristiques. Et j'ai ajouté une information supplémentaire pour chaque noeud pour l'entraînement: la classe. Pour notre problème c'est une segmentation donc c'est soit "ce n'est pas de la tumeur : 0" , "c'est de la tumeur: 1". Donc il y a 2 classes uniquement. Et on récupère cette information grâce au masque fournit dans le dataset. 

C'est ici que j'ai eu un sérieux goulot d'étranglement. Pour charcher 1 seule IRM, cela me prenait 2 minutes, ce qui fait un total de 7h par exemple si on devait charger tout le dataset de BRATS avec mon laptop. 

J'ai tenté d'optimiser mon extraction des features ensuite en faisant cela :
Je voulais réduire les overheads donc j'ai utilisé une boîte anglobante autour de la tumeur pour ne pas traiter tout le volume.
J'ai utilisé un dictionnaire sp_index_map (superpixel -> liste de voxels)  pour ne pas avoir à calculer superpixels == sp_id (utilisé pour construire le masque booleen) à chaque fois. 
J'ai voulu faire des batch de 50 pour accélérer le tout grâce à Joblib. Et j'ai exploré la fusion des supervoxels pour accélérer encore plus le tout. 

Sinon j'avais une autre idée potentiellement interessante : lisser l'image partout sauf sur la tumeur grâce aux contourages des tumeurs. Et par exemple refaire SLIC là dessus et donc avoir des supervoxels très détaillées  (donc amas de petits supervoxels) sur les tumeurs et des énormes supervoxels sur les textures homogènes comme des tissus du cerveau ou encore le crâne. J'ai également exploré la possibilité d'utiliser un octree (quadtree étendue à la 3D) afin d'avoir un graphe potentiellement encore mieux. Je pense que cela pourrait être une bonne alternative à SLIC. 

##### Création des graphes
L'article construit ses RAG (Region Adjency Graph) en connectant les noeuds (supervoxels) qui sont adjacents spatialement. 
J'ai donc adopté la même approche. Je traîte chaque coordonnée x y z pour récupérer le supervoxel courant. 
Je trouve les coordonnées voisines avec l'élément structurant 6-voisinage 3D (x\pm1,y,z), (x,y\pm1,z), (x,y,z\pm1)
Ensuite si le voisin appartient à un supervoxel `neighbor_sp` différent, je créer une arête 

Ensuite y a une étape cruciale de suppression des doublons car on peut ajouter des arêtes inutiles vu qu'on traîte tout dans l'ordre. 
Peut-être qu'il y avait plus efficace comme implémentation pour le RAG.

## Définition du GAT
Je me suis inspiré de l'article pour faire ma propre version de leur modèle. J'ai utilisé la librairie pyg (pytorch geometric), c'est le must have pour quelqu'un qui veut faire des gnns. J'ai choisi de faire quelque chose de plutôt simple car de toute façon je ne peux pas tester vu que je n'ai pas de machine assez puissantes. 

J'ai mis 2 couches GATConv grâce à la librairie pyg. 
Une multi-head attention sur la première couche suivi d'une LeakyReLU comme fonction d'activation. 
Je suis un peu sceptique sur l'absurde facilité que j'ai eu à écrire cette classe pour le modèle, je pense que ça ne fonctionnera jamais. 

Explication de ce qu'il se cache derrière GATCONV de pyg :

c'est la couche Graph Attention Network pour faire simple, au lieu de sommer/moyenner/accumuler les informations des noeuds voisins, comme le ferait un GNN convolutif (GCN), on va pondérer chaque voisin avec un **score d'attention**. Ainsi les noeuds les plus pertinents ont plus de poids dans la mise à jour de notre noeud central (c'est tout le principe de l'article). 

Sauf que vu que on calcul pas mal de scores, on va ajouter une multi-head attention pour avoir plusieurs têtes d'attention parallèlement, l'avantage c'est que réseau va apprendre grâce à cela quelles arêtes sont pertinentes et lesquelles ne le sont pas. 

Au final GATCONV prend en entrée les 20 features des noeuds, la liste des arêtes et renvoie les nouvelles features updates de chaque noeud après application du mécanisme GAT. 

Ils utilisent des couches GATConv avec des dimensions croissantes des **features cachées** à chaque couche (par exemple, 20 → 1024 → 1280 → 768, etc ..), ce qui montre l’importance de l’expansion progressive pour apprendre des représentations complexes dans un graphe dense.
J'ai pris le choix de fixer le nombre de dimensions cachées à 32 pour ne pas être trop complexe mais être quand même assez discriminant pour les glioblastomes. Avec 32 dimensions, chaque head-attention gère des vecteurs de 8 dimensions. On peut utiliser grid-search pour optimiser cela.

## Entraînement
Pour entraîner, j'ai repris la configuration de l'article à savoir l'optimiser Adam avec un scheduler qui contrôle le learning rate de manière automatique pendant l'apprentissage avec un LR au départ égal à 0.99 , "exponential decay at the rate of 0.00001".
Eux ils se sont permis le luxe d'avoir des mini-batchs de 6 graphes (chaque graphe est une IRM (ensemble de supervoxels))

La fonction de perte est logiquement une CrossEntropyLoss car c'est une segmentation 2 classes. 

J'ai mis un setup de checkpointing pour sauvegarder les poids de temps en temps, 

L'entraînement leur a pris 2 jours sur cette bécane : AMD Ryzen 7 4800HS 2.90 GHz processor connected to Google Colab with a Tesla K80 GPU having 2496 CUDA cores and 35 GB of DDR5 VRAM. 

J'essayerai d'optimiser tout le code et le modèle le jour où j'aurais accès à une machine de compète. Néanmoins c'est interessant de chercher des compromis pour optimiser pour penser aux configurations plus petites.  

## Related work GAT

**Exploring graph-based neural networks for automatic brain tumor segmentation,” in Intl. Symposium: From Data to Models and Back, pp. 18–37, Springer, 2020.**

Cet article est un survey interessant sur l'utilisation des GNNs pour la segmenter des tumeurs. Il critique les coûts computationnels des GCN et leur incapacité à capter des relations globales en 3D. 
Plusieurs variantes de GNN sont testées : **GrapheSAGE**, **GCN**, **GAT**.
**Ils ont parlé d'explicabilité ! Aleluya ** Ils utilisent **SHAP** pour comprendre les contributions des différentes modalités d'IRM dans les prédictions.
GrapheSAGE obtient les meilleurs résultats (dice scores les plus élevés).
Les GCN sont plus rapides à  entraîner que les CNN. C'est interessant. 
Bien que les GCN n'égalisent pas les CNN (genre U-NET), c'est une alternative qui est efficace pour des ressources plus limitées. 

**A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation. https://doi.org/10.1007/978-3-031-08999-2_30**
Cet article présente un modèle hybride combinant un GNN avec un CNN pour la segmentation des tumeurs, c'est assez original. Le rôle du CNN est ici d'affiner les prédictions du GNN en intégrant des détails locaux. Ils l'ont utilisé sur brats2021 et les résulats montrent une améliorationde 2% des scores de Dice médians par rapport à une approche GNN seule. 

**A novel deep learning graph attention network for Alzheimer’s disease image segmentation https://doi.org/10.1016/j.health.2024.100310**
Cet article propose une méthode de segmentation U-GAT qui combine un UNET avec un GAT. Cela exploite la capacité du GAT à capturer les relations complexes entre les pixels tout en bénéficiant des mécanismes d'attention et d'upsampling propres aux UNET. Ils l'ont utilisé sur des images microscopiques de cellules neuronales associées à la maladie d'alzheimer. Cela surpasse U-Net, SegNet, VGG16, GAT. (accuracy de 86.5%, F1-score 0.719). 

**TGAP-Net: Twin Graph Attention Pseudo-Label Generation for Weakly Supervised Semantic Segmentation 10.1109/JBHI.2025.3525647**
J'ai pas accès à l'article mais cela a l'air novateur. C'est dans la même vibe multi-class segmentation. Ils utilisent un GAT, et une fonction d'aggrégation "Global Classified Max Pooling". 
Le GCMP améliore la transmission des signaux ce qui augmente la précision pour la classification des segmentées, ce qui est pertinent pour du multi-class. Cela améliore notamment la segmentation par exemple sur les nécroses (quand la tumeur est "vieille" et que son centre devient mort et donc noir, ça cause parfois des problèmes aux modèles de segmentation qui ne gardent pas la nécrose dans la segmentation). 

**UnSegMedGAT: Unsupervised Medical Image Segmentation using Graph Attention Networks Clustering https://doi.org/10.48550/arXiv.2411.0196**
Utilise GAT avec les ViT (vision transformers). La fonction de perte est basée sur la modularité -> mesure utilisée en théorie des graphes pour évaluer la qualité d'une partition en clusters au sein d'un graphe. Cela quantifie à quel point les noeuds d'un cluster sont densément connectés entre eux par rapport à des connexions aléatoires dans le graphe. modularité élevé -> les noeuds dans le même cluster sont fortement connectés et les connexions entre cluster sont faibles. Pour de la segmentation, maximiser la modularité revient à trouver une partition de l'image via les noeuds de notre graphe où les pixels similaires sont regroupés. 
**Ils s'appuient sur des caractéristiques pré-entraînées par un ViT pour segmenter les images sans annotation.** 
Ils font leurs tests sur le jeu de données **ISIC-2018** et **CVC-ColonDB**.
L'intérêt de leur approche c'est que c'est non supervisé et cela surpasse d'autres méthodes non supervisées comme DSM et ViT-KMeans et rivalisant avec MedSAM. 
**DSM (Deep Spectral Methods)** : segmentation sémantique non supervisée via méthodes spectrales. 
**ViT-KMeans** : Pour extraire des caractéristiques riches dans les images et appliquer un clustering pour de la segmentation
**MedSAM (Segment Anything Model in medical)** : basé sur les ViT, pretrain sur un vaste corpus d'images médicales variées. Besoin de bcp de données annotées pour l'utiliser, et puis c'est d'avantage performant sur des données similaires à celles de la base d'entraînement. 

## Related Work Super-voxels 
**Supervoxel Segmentation with Voxel-Related Gaussian Mixture Model 10.3390/s18010128**
Présente une méthode de segmentation de supervoxels dans des vidéos basées sur un mélange de gaussiennes lié aux voxels ("Voxel-Related Gaussian Mixture Model, VR-GMM"). Principe : segmente **deux cadres consécutifs** en même temps pour garantir une **cohérence temporelle**  tout en limitant l'explosion computationelle. 
Chaque supervoxel est représenté par deux distributions gaussiennes partagées par des paramètres de couleur, adaptées aux variations spatiales. Ils font une optimisation de l'algorithme de EM (Expectation-Maximization) ce qui permet de garantir que les supervoxels sont homogènes en taille qui restent fidèles aux limites des objets, restant robustes fasse aux mouvements et autres intercations. 

**Graph-based Supervoxel Computation from Iterative Spanning Forest https://hal.science/hal-03171076v1** 
Méthode assez innovante et originale pour avoir des bons supervoxels. Ils l'appelent "ISF2SVX (Iterative Spanning Forest for Supervoxels)".
Ils utilisent les ISF : forêts couvrantes itératives et l'algorithme IFT (Image Foresting Transform). On peut contrôler le nombre de supervoxels comme dans SLIC. C'est potentiellement utile pour de la segmentation sémantique de vidéos. 
ISF -> Extension d'IFT qui ajoute un processus itératif pour améliorer les résulatsq de IFT en plusieurs étapes (élimination des mauvaises seed)
IFT -> basé sur la théorie des graphes, utilise des seeds  (pixels de départs) qui servent de points initiaux pour créer des arbres
A partir de ces graines, l'algo explore les chemins optimaux entre les noeuds voisins et on peut minimiser selon plusieurs critères comme l'intensité, la distance spatiale, etc .. 

**Octree Representation Improves Data Fidelity of Cardiac CT Images and Convolutional Neural Network Semantic Segmentation of Left Atrial and Ventricular Chambers https://doi.org/10.1148/ryai.2021210036**
Enfin des chercheurs qui utilisent les octrees pour l'imagerie médicale. Une représentation basée sur les octrees, optimisée pour l’imagerie médicale. L’objectif est de réduire drastiquement l’empreinte mémoire des volumes 3D tout en préservant la qualité de l’image et des frontières des structures segmentées. Ils introduisent un paramètre de **tolérence d'intensité** pour contrôler le niveau de compression ce qui a un impact majeur sur la granularité des cubes de l'octree. 
Ils proposent OctNet, un réseau convolutif 3D pour traiter directement les volumes 3D compressées avec les octrees. Opère sur la résolution native ce qui est bien donc ça réduit les coûts en mémoire et reste fidèle aux frontières anatomiques. On pourrait utiliser leur approche pour notre GAT au lieu d'utiliser SLIC.