Guide rapide:

	I: Compilation: (avec GCC standard gnu99)
		- ligne de commande:	gcc -Wall som.c -O3 -o som -lm -std=gnu99
		- make:		make
	II: Usage:
		- default: ./som <fichier>
		
	III: A propos des dataset:
		Data set iris from kaggle:
		https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download
		Data set penguins from kaggle: (modifié: retrait de: island, sex, année & label en fin de vecteur)
		https://www.kaggle.com/datasets/larsen0966/penguins
