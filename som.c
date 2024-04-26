/* Nom           : som.c
 * Role          : Implementation de Self-Organizing Map de Kohonen sur dataset IRIS
 * Version       : 13/06/2022
 * Licence       : L3-Intelligence Artificielle et Apprentissage
 * Compilation   : standard: gnu99
 *                 gcc -Wall som.c -O3 -o som -lm -std=gnu99
 * Usage : Necessite le dataset csv: ./som "file.csv" */    
/*******************************************************************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
/*******************************************************structrures*************************************************************************************/
// linked list structure pour enregistrer les BMU egaux:
typedef struct List {
    int x;
    int y;
    struct List *next;
} List, *list;


// Structure des frames de donnees
typedef struct Data Data;
struct Data{
    double *W, CNorm;
    char *Label;
};

// Structure des donnees a traiter
typedef struct DataP DataP;
struct DataP{
    Data *OriginalSet, *NormalizedSet;
    int Height, Width, *CallIdx;
    double *VMean;
};

// Element constitutif de la SOM
typedef struct SOM SOM;
struct SOM{
    double *W, DEuclid;
    char *Label;
};

/***************************************************Functions prototypes********************************************************************************/
void usage(char *);
list cons(int, int, list);
DataP LoadParameter(char *);
Data RecordData(char *, int);
Data NormalizeW(Data, int);
double* KMean(DataP);
int CountLineCSV(char*);
int* rectangle(int);
void GenMap(int, int, SOM[*][*], int, DataP);
double RandomWeigth(double, double);
int* ShuffleVect(int[], int);
double DEuclid(int, double[*], double *);
void Bmu(int, int, SOM[*][*], int *, int, double*);
void Learn(double, int, int*, int, int, SOM[*][*], double*, int); 
void label(int, int, SOM[*][*], DataP);
void BestZone(int, int, SOM[*][*], DataP);
void Autotest(int, int, SOM[*][*], DataP); 

/**********************************************************Main*****************************************************************************************/
int main(int argc, char * argv[]){
//  Parametre de base: alpha proche de 1, Ninit (voisinage) entre 50 et 30% des neurones, 2 phases 1: 1/5 2: 2/5, pour la 2 500 iterations par features.
    double AInit = 0.7, ACurrent, CNinit = 0.5;
    if(argc != 2) usage("The som program requires a dataset in its current directory! ./som dataset\n");
    srand(time(NULL));
    char *file = argv[1];
    DataP DataParameter = LoadParameter(file);
    DataParameter.VMean = KMean(DataParameter);

//- N Nombre de neurones: D'apres Jing Tian et al. (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.654.8502&rep=rep1&type=pdf): N ≈ 5sqrt(DataP.Height),arrondit pour avoir une matrice carre n*n de n^2 neurones
    double N = 5*sqrt(DataParameter.Height);
//- dim liste de deux dimensions pour un tableau rectangulaire [long, larg]
    int *dim = malloc(sizeof(int)*2);
    if (! dim) usage("dim: manque de RAM");
    dim = rectangle(N);
    SOM Map[dim[0]][dim[1]];

// - initialisation des connexions montantes wij dont les valeurs sont generees aleatoirement: faible entre 0.1 et 0.5 ou autour de la moyenne des composantes des vecteurs entre xi +/- 0.2 a 0.5
    GenMap(dim[0], dim[1], Map, DataParameter.Width, DataParameter);  
    int Ninit = floor(sqrt(CNinit*N)), Nsize=0, Nfin=1, TourMax = (500*DataParameter.Width)/0.8;
    DataParameter.CallIdx = ShuffleVect(DataParameter.CallIdx, DataParameter.Height);
//  Boucle principale:
    for(int t=0; t<TourMax; t++){
//      -distance de propagation  version simplifie Kohonen Nsize = Ninit(1-t/Ttotal), coefficient d'apprentissage h *1 ou *0.
        ACurrent = AInit * (1-(double) t/TourMax);
        Nsize = lround((Ninit)*(1-(double) t/TourMax));
//      Taille voisinage minimal
        if(Nsize<Nfin) Nsize = Nfin;
//      presentation des vecteurs de donnees:
        for(int a=0; a<DataParameter.Height; a++){
            int i = DataParameter.CallIdx[a], BMU[2] = {0, 0}, *PBMU = BMU;
            double *wij = DataParameter.NormalizedSet[i].W; 
//          Calcul BMU et Maj des distances Euclidiennes */       
            Bmu(dim[0], dim[1], Map, PBMU, DataParameter.Width, wij);
            Learn(ACurrent, Nsize, PBMU, dim[0], dim[1], Map, wij, DataParameter.Width);
        } 
    }
    label(dim[0], dim[1], Map, DataParameter);
//  A partir d'ici la som est entrainée et étiquetée
    BestZone(dim[0], dim[1], Map, DataParameter);
    Autotest(dim[0], dim[1], Map, DataParameter);
    return 0;
}

/***************************************************auxiliary functions*********************************************************************************/
/* Retourne un message d'erreur et sort du programme:
   Parametres: char *<message d'erreur> */
void usage(char * message) {fprintf(stderr, "%s\n", message); exit(1);}

/* constructeur de linked list:
   Parametres: int <valeur de x>, int <valeur de y>, List* <pointeur sur next ou NULL> */
list cons(int v1, int v2, list L){
    list new = malloc(sizeof(List));
    if (! new)  usage("Problem with list!\n");
    new -> x = v1;
    new -> y = v2;
	new -> next = L;
	return new;
}

/*  Extrait les informations d'un dataset dans une fichier csv de ligne de x arguments termine par un label:
    Parametres: char *<Nom_du_fichier> */
DataP LoadParameter(char *file){ 
    FILE *f = fopen(file, "r");
    if (! f)  usage("Problem with the dataset file!\n");
    char CurrentChar, Buffer[1000];
    int NFeat=0, NData = CountLineCSV(file), i=0;
//  Lire et enregistrer le dataset
    DataP DataPtemp;
    DataPtemp.OriginalSet = (Data*)malloc(sizeof(Data)*NData);
    DataPtemp.NormalizedSet = (Data*)malloc(sizeof(Data)*NData);
//  Determiner le nombre de features
    while((CurrentChar=fgetc(f)) != '\n'){
        if (CurrentChar == ',') NFeat++;
    }

    while(fgets(Buffer, 1000, f)){
        Buffer[strlen(Buffer)-2] = '\0';
        DataPtemp.OriginalSet[i] = RecordData(Buffer, NFeat);
        DataPtemp.NormalizedSet[i] = NormalizeW(DataPtemp.OriginalSet[i], NFeat);
        i++;
    }
    fclose(f);

    DataPtemp.Height = NData;
    DataPtemp.Width = NFeat;
    DataPtemp.CallIdx = (int*)malloc(sizeof(int)*NData);
//  Initialisation de l'index d'appel
    for(int x=0; x<NData; x++) DataPtemp.CallIdx[x] = x;
    return DataPtemp;
}

/* Decoupe une frame et l'enregistre 
   Parametres: char *<Nom_du_fichier>, int <Nombre_d_arguments> */
Data RecordData(char *data, int Nargs){
    Data Frame;
    Frame.W = (double*)malloc(sizeof(double)*Nargs);
    if (! Frame.W) usage("RecordData error!\n");
    int i = 0;
    char * args = strtok(data, ",");
    while(args != NULL){
        if(i<Nargs)Frame.W[i] = strtod(args, NULL);
        else{
            Frame.Label = (char*)malloc(sizeof(char)*strlen(args));
            strcpy(Frame.Label, args);
        }
        args = strtok(NULL, ",");
        i++;
    }
    return Frame;
}

/* Norme d'un vecteur: V(xi^2+...+zi^2) 
   Parametres: Data <frame_de_donnees>, int <nombre_d_argulents> */
Data NormalizeW(Data W,int Nargs) {
    Data Frame;
    Frame.W = (double*)malloc(sizeof(double)*Nargs);
    if (! Frame.W) usage("NormalizeW error!\n");
//  calcul du coefficient de normalisation
    double norm=0;
    for(int i=0; i<Nargs; i++) norm += pow(W.W[i],2);
    norm = sqrt(norm);
    Frame.CNorm = norm;  
//  Normalisation 
    for(int i=0; i<Nargs; i++) Frame.W[i] = W.W[i]/norm;
//  label
    Frame.Label = (char*)malloc(sizeof(char)*strlen(W.Label));
    strcpy(Frame.Label, W.Label);
    return Frame;
}

/* Calcul du vecteur moyen:
    parametre Datap <structure data>*/
double *KMean(DataP DataP){
    double *VMean;
    VMean = (double*)malloc(sizeof(double)*DataP.Height);
    for(int x=0; x<DataP.Width; x++){
        double ArgMean = 0;
        for(int y=0; y<DataP.Height;y++) ArgMean += DataP.NormalizedSet[y].W[x];
        VMean[x] = ArgMean/DataP.Height;
    }
    return VMean;
}

/*  compte le nombre de ligne d'un fichier csv en retirant la ligne de przsentation 
    Parametres: char *<nom_du_fichier>*/
int CountLineCSV(char *file){ 
    FILE *p = fopen(file, "r");
    char Pass[1000];
    int Count = 0;
    while(fgets(Pass, 1000, p)) Count++;
    fclose(p);
    return Count-1;
}

/* Calcul le rectangle avec 1 dimension d'écart pour contenir 5 racince carré de N éléments
    Parametres: int nombre de ligne du dataset */
int* rectangle(int number) {
    int * result = malloc(sizeof(int)*2);
    if (! result) usage("result : manque de RAM");
    int median = (int) ceil(sqrt(number))/2, x = median, y = median;
    while (x*y < number) {
        if (x == y)  x = x + 1;
        else y = y + 1;
    }
    result[0] = x;
    result[1] = y;
    return result;
}

/* Genere les vecteurs de base de maniere aleatoire:
   - faible entre 0.1 et 0.5  ou dans un intervalle restreint autour de la moyenne des composantes des vecteurs de donnees, entre xi + ou - 0.2 a 0.5 */
void GenMap(int lon, int larg, SOM Map[lon][larg], int Nargs, DataP DataP){
    for(int i=0; i<lon; i++){
        for(int j=0; j<larg; j++){
            Map[i][j].W = (double*)malloc(sizeof(double)*Nargs);
            if (! Map) usage("Map : manque de RAM");
            //for(int z=0; z<Nargs; z++) Map[i][j].W[z] = RandomWeigth(0.1, 0.5); // poids faible
            for(int z=0; z<Nargs; z++) Map[i][j].W[z] = DataP.VMean[z] - RandomWeigth(-0.3, 0.3); // poids relatif au vecteur moyen
        }
    }
}

/* Generateur aleatoire de double entre deux bornes;
   Parametres: double <borne_inferieur>, double <borne_superieur> */
double RandomWeigth(double start, double end){
    return start + (double) rand()/((double)RAND_MAX/(end-start));
}

/* melange un vecteur d'entier
   Parametres: int[] <vecteur d'entier>, int <nombre d'elements>*/
int* ShuffleVect(int vector[], int sizeVector){
    int tmp, random;
    int * indexAleatoire = malloc(sizeof(int)*sizeVector);
    if(! indexAleatoire) usage("Problem with the dataset file!\n");
    for(int i=0;i<sizeVector; i++) indexAleatoire[i] = vector[i];
    for(int i=0; i<sizeVector; i++){
        random = (rand()/ (RAND_MAX  / sizeVector));
            tmp = indexAleatoire[i];
            indexAleatoire[i]=vector[random];
            indexAleatoire[random] = tmp;
        }
    return indexAleatoire;
}

/* Distance Euclidienne sur x arguments :√( (xwij -xMapcarre)^2 + ... +(zwij -zMapcarre)^2  )
   Parametres: int <dimenstion_tableau>, double[*] <tableau>, double* <pointeur sur tableau> */
double DEuclid(int n, double* W, double * wij) {
    double Deuc=0;
    for(int x=0; x<n;x++) Deuc += pow(wij[x]-W[x], 2);
    return sqrt(Deuc);  
}

/* Recherche du BMU
   Parametres: int <dimension>, SOM[n][n] <SOM>, int*[2] <coordonnee du BMU> */
void Bmu(int lon, int larg, SOM Map[lon][larg], int *BMU, int Width, double *wij){
    int Bmu_size = 1;
    list Bmus = NULL;
    Bmus = cons(0, 0, Bmus);
    for(int i=0; i<lon; i++){
        for(int j=0; j<larg; j++){
            Map[i][j].DEuclid = DEuclid(Width, Map[i][j].W, wij);
            /* test */
            //printf("BMU[%i][%i] : %f - Map[%i][%i] : %f\n", Bmus->x, Bmus->y, Map[Bmus->x][Bmus->y].DEuclid, i, j, Map[i][j].DEuclid);
            if(Map[Bmus->x][Bmus->y].DEuclid == Map[i][j].DEuclid) {
                Bmu_size++;
                Bmus = cons(i, j, Bmus);
            }
            else if(Map[Bmus->x][Bmus->y].DEuclid>Map[i][j].DEuclid) {
                Bmu_size = 1;
                Bmus = NULL;
                Bmus = cons(i, j, Bmus);     
            }
        }
    }
    if (Bmu_size > 1) {
        int current = 0, winner = rand()%Bmu_size; 
        while(current != winner){
            Bmus = Bmus->next;
            current++;
        }
    }
    BMU[0] = Bmus->x;
    BMU[1] = Bmus->y;
}


/* Propagation du WTA sur les voisins a Nsize porte via les 8 conenxions laterales.
   Parametres: int <iteration>, int <max d'iteration>, double <coefficient d'apprentissage>, int <taille du voisinage>, int* <coordonees>, int <dimension>, SOM[][] <SOM, double <vecteur>, int <nombre de features>  */
void Learn(double AlphaCurrent, int Nsize, int* PBMU, int lon, int larg, SOM Map[lon][larg], double* wij, int Nargs) {
//  détermination de la zone de propagation:
    int start_long = 0, end_long = lon, start_larg = 0, end_larg = larg;
    if (PBMU[0] - Nsize > 0) start_long = PBMU[0] - Nsize;
    if (PBMU[0] + Nsize < lon) end_long = PBMU[0] + Nsize +1;
    if (PBMU[1] - Nsize > 0) start_larg = PBMU[1] - Nsize;
    if (PBMU[1] + Nsize < larg) end_larg = PBMU[1] + Nsize +1; 
//  apprentissage J* et voisingage:
    for(int x=start_long; x<end_long; x++){
        for(int y=start_larg; y<end_larg; y++){
            for(int z=0; z<Nargs; z++) Map[x][y].W[z] = Map[x][y].W[z] + AlphaCurrent*(wij[z]-Map[x][y].W[z]);
        }
    }
}

/* fonction de distribution de label en fonction du bmu pour chaque neurone:
   Parametres: int <longueur>, int <largeur>, SOM <matrice des neurones>, DataP <information du dataset> */
void label(int lon, int larg, SOM Map[lon][larg], DataP DataP){
    for(int x=0; x<lon; x++){
        for(int y=0; y<larg; y++){
            float DEuclidmin = 100;
            int index =0;
            for(int z=0; z<DataP.Height; z++){
                float DEuclid=0;
                for(int w=0; w<DataP.Width; w++) DEuclid += pow(Map[x][y].W[w]-DataP.NormalizedSet[z].W[w], 2);
                if(DEuclid < DEuclidmin){
                    DEuclidmin = DEuclid;
                    index = z;
                }
            }
            Map[x][y].Label = DataP.OriginalSet[index].Label; 
        }
    }
}

/* Fonction de representation final des contenues des "neurones" et d'une carte de "clustering" en comparant les vecteurs avec ceux du dataset pour trouver lequel match le mieux avec quel classe.0
   Parametres: int <dimension>, SOM <matrice>, DataP <dataset> */
void BestZone(int lon, int larg, SOM Map[lon][larg], DataP DataP){
//  Recuperation des labels:
    char LabelBuffer[1000] = "";
    int NLabel = 0, i = 0;
    for(int x=0; x<DataP.Height; x++){
        if(strstr(LabelBuffer, DataP.OriginalSet[x].Label) == NULL){
            strcat(LabelBuffer, DataP.OriginalSet[x].Label);
            strcat(LabelBuffer, ", ");
            NLabel ++;
        }     
    }
    char *Label[NLabel], *Labs = strtok(LabelBuffer, ", ");
    while(Labs != NULL){
        Label[i] = Labs;
        Labs  = strtok(NULL, ", ");
        i++;
    }

//  Legende:
    printf("Legende:\n");
    for(int x=0; x<NLabel; x++) printf("%d %s\n", x, Label[x]);

//  representation de la SOM
    printf("\n");
    for(int x=0; x<lon; x++){
        for(int y=0; y<larg; y++){
            // Etiquetage
            for(int w=0; w<NLabel; w++){
                if(strcmp(Map[x][y].Label, Label[w]) == 0) printf("%d ", w);
            }
        }
        printf("\n");
    }
}

/* fonction d'evaluation de la som entrainee pour verifier qu'elle est capable de retrouver les labels des vecteurs du set de donnees */
void Autotest(int lon, int larg, SOM Map[lon][larg], DataP DataP) {
    int right = 0;
    for(int x=0; x<DataP.Height; x++){
        int  BMU[2] = {0, 0}, *PBMU = BMU;
//      Maj des distances Euclidiennes:
//      recherche du bmu pour connaitre le neurone qui reconnait le vecteur:
        Bmu(lon, larg, Map, PBMU, DataP.Width ,DataP.NormalizedSet[x].W);
//      verification si la catégorie est la même:
        if(strcmp(Map[BMU[0]][BMU[1]].Label, DataP.NormalizedSet[x].Label) == 0) right++;
    }

    printf("\nPertinence de classification: %f%% \n", (double) right/DataP.Height*100);
}
