# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import mdshare
import argparse
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description = "Skrypt zaliczeniowy - rzutowanie tensora na przestrzen trojwymiarowa.")
parser.add_argument('-d' , '--dimensions', type = int, default = 3, help = "Liczba wymiarow do ktorych zostanie zredukowany tensor. Domyslnie: 3 ")
parser.add_argument('-s' , '--step', type = int, default = 500, help = "Parametr ustawia probkowanie. Reszta danych zostanie zignorowana. Domyslnie: 500 ")
parser.add_argument('-p' , '--perplexity', type = float, default = 30., help = "Parametr zwiazany z liczba najblizszych sasiadow probki, zostana uzyte do algorytmu uczenia sie. Im wiekszy zbior danych tym zalecana wyzsza zlozonosc. Zalecany zakres od 5 do 50. Domyslnie: 30")
parser.add_argument('-l' , '--learning_rate', type = float, default = 100., help = "Wspolczynnik uczenia sie algorytmu. Jesli wspolczynnik jest zbyt wysoki dane moga zostac skupione, wygladac jak 'kula'. W przypadku zbyt niskiego parametru dane beda rozproszone, z kilkoma punktami o innej wartosci. Zalecany przedzial od 10 do 1000. Domyslnie: 200 ")
parser.add_argument('-n' , '--n_iter',  type = int, default = 1000, help = "Maksymalna liczba iteracji w oczekiwaniu na najlepsza optymalizacje. Zalecana wartosc powyzej 250. Domyslnie: 1000")
parser.add_argument('-v' , '--verbose',  type = int, default = 0, help = "Poziom szczegolowosci. Domyslnie: 0")
parser.add_argument('-a' , '--angle', type = float, default = 0.5, help = "Kompromis miedzy szybkoscia, a dokladnoscia w przypadku algorytmu T-SNE Barnes'a - Hut'a. Metoda ta jest malo wrazliwa na zmiany tego parametru w zakresie 0,2 - 0,8. Kat mniejszy niz 0,2 zwieksza czas obliczen, a kat wiekszy niz 0,8 ma wiekszy blad obliczen. Domyslnie: 0,5")


#przypisywanie wartosci argumentow do zmiennych
args = parser.parse_args()
Dimensions = args.dimensions
Step = args.step
Perplexity = args.perplexity
Learning_rate = args.learning_rate
N_iter = args.n_iter
Verbose = args.verbose
Angle = args.angle


#pobieranie danych
dataset = mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
with np.load(dataset) as f:
    X = np.vstack([f[key] for key in sorted(f.keys())])

#w zaleznosci od ilosci wymiarow wybrany zostanie inny tryb wyswietlania
if Dimensions == 3:
    
    Y = TSNE(n_components=Dimensions, perplexity = Perplexity , learning_rate = Learning_rate , n_iter = N_iter, verbose = Verbose, angle = Angle).fit_transform(X[::Step])
    
    Y[:, 0] = np.interp(Y[:, 0], (Y[:, 0].min(), Y[:, 0].max()), (-np.pi, np.pi)) #skalowanie danych do zakresu od -pi do +pi
    Y[:, 1] = np.interp(Y[:, 1], (Y[:, 1].min(), Y[:, 1].max()), (-np.pi, np.pi)) 
    Y[:, 2] = np.interp(Y[:, 2], (Y[:, 2].min(), Y[:, 2].max()), (-np.pi, np.pi))


    plt.scatter(Y[:, 0], Y[:, 1], c=Y[:, 2], s = 15, alpha = 0.8 , cmap = 'viridis' ) 
    plt.xlim(-np.pi, np.pi) #ustawienie wartosci osi wykresu od -pi do +pi
    plt.ylim(-np.pi, np.pi) 
    plt.xticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) 
    plt.yticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) 
    plt.axis('square') # wykres w ksztalcie kwadratu 
    cbar=plt.colorbar() #trzeci wymiar prezentowany jest jako barwa punktu - ustawienie odpowiednie parametrow wyswietlania
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels(['-π', 0, 'π']) 
    
    plt.show() 
    
    print("Done") #zakonczenie programu
elif Dimensions == 2:
    
    Y = TSNE(n_components=Dimensions, perplexity = Perplexity , learning_rate = Learning_rate , n_iter = N_iter, verbose = Verbose, angle = Angle).fit_transform(X[::Step])
    
    Y[:, 0] = np.interp(Y[:, 0], (Y[:, 0].min(), Y[:, 0].max()), (-np.pi, np.pi)) #skalowanie danych do zakresu od -pi do +pi
    Y[:, 1] = np.interp(Y[:, 1], (Y[:, 1].min(), Y[:, 1].max()), (-np.pi, np.pi)) 

    plt.scatter(Y[:, 0], Y[:, 1], s = 15, alpha = 0.8 , cmap = 'viridis') 
    plt.xlim(-np.pi, np.pi) #ustawienie wartosci osi wykresu od -pi do +pi
    plt.ylim(-np.pi, np.pi) 
    plt.xticks([-np.pi, 0, np.pi], ['-π', 0, 'π'])
    plt.yticks([-np.pi, 0, np.pi], ['-π', 0, 'π'])
    plt.axis('square') # wykres w ksztalcie kwadratu 

    plt.show() 

    print("Done") #zakonczenie programu
    
else:
    print("Niestety, program napisany jest do redukcji tensora tylko do przestrzeni dwu- lub trojwymiarowej")
    # ograniczenie wprowadzone przez algorytm Barnes'a-Hut'a. Wybranie większej ilosci wymiarow skutkuje wyswietleniem tego komunikatu