import numpy as np
import pandas as pd
import sys
import os
import json
from collections import Counter
import glob
import networkx as nx

#sys.path.insert(1, '/home/chacoma/librerias/python')
#from plotter_lib import *

db = '/home/chacoma/Lineas/football_4/data/'
arx1 = db+'Sample_Game_1/Sample_Game_1_RawEventsData.csv'
arx2 = db+'Sample_Game_2/Sample_Game_2_RawEventsData.csv'

info = {
    "game1": {"team1": {"per1": 0, "per2": 1, "arquero": "Player11", "localia": "Home"}, "team2": {"per1": 1, "per2": 0, "arquero": "Player25", "localia": "Away"}, "arx": arx1},
    "game2": {"team1": {"per1": 1, "per2": 0, "arquero": "Player11", "localia": "Home"}, "team2": {"per1": 0, "per2": 1, "arquero": "Player25", "localia": "Away"}, "arx": arx2}
}


def get_redes():

    # pars
    game = "game2"
    team = "team2"
    nlinks = 50

    dbs = "%s/%s/" % (game, team)
    if not os.path.exists(dbs):
        os.makedirs(dbs)

    localia = info[game][team]["localia"]
    arx = info[game]["arx"]
    df = pd.read_csv(arx)

    red = {}
    Links = []
    X, Y = [], []
    flag = 1

    for index, row in df.iterrows():

        if row["From"] != info[game][team]["arquero"] and row["To"] != info[game][team]["arquero"]:

            # ARMADO DE RED ------------------------------------
            if row["Type"] == "PASS":

                if localia == row["#Team"]:

                    # nodos de la red
                    link = "%s_%s" % (row["From"], row["To"])

                    try:
                        red[link] += 1
                    except:
                        red[link] = 1

                    Links.append(link)

                    # Para Centroides
                    X.append(row["Start X"])
                    Y.append(row["Start Y"])

                    if len(Links) > nlinks:

                        a_sacar = Links[0]
                        Links.pop(0)
                        red[a_sacar] -= 1

                        if red[a_sacar] == 0:
                            red.pop(a_sacar)

                        X.pop(0)
                        Y.pop(0)

            # BUSQUEDA DE SHOTS ---------------------------------
            if row["Type"] == "SHOT" and len(Links) > 49:

                # chequeo si el mean del centroide es del lado izq o der de la cancha
                xcm = np.mean(X)

                if xcm < 0.5:
                    xp = 0
                else:
                    xp = 1

                # busco de que lado de la cancha jugaba el equipo en el periodo dado
                if row["Period"] == 1:
                    lado = info[game][team]["per1"]
                else:
                    lado = info[game][team]["per2"]

                # si el xp es distinto al lado entonces el equipo esta jugando adelantado
                if xp == lado:
                    adelantado = 0
                else:
                    adelantado = 1

                # Shot a favor y con eq adelantado
                if localia == row["#Team"] and adelantado:

                    print("A favor", row["Period"], xcm)

                    arxs = dbs+"S1_%d.json" % int(row["Start Time [s]"])
                    json.dump(red, open(arxs, 'w'))

                # Shot en contra y con eq no adelantado
                if localia != row["#Team"] and not adelantado:

                    print("En contra", row["Period"], xcm)
                    arxs = dbs+"S0_%d.json" % int(row["Start Time [s]"])
                    json.dump(red, open(arxs, 'w'))

                    """
        # Shot a favor y con eq adelantado
        if localia == row["#Team"]:

            print("A favor", row["Period"])

            arxs = dbs+"S1_%d.json" % int(row["Start Time [s]"])
            json.dump(red, open(arxs, 'w'))

        # Shot en contra y con eq no adelantado
        if localia != row["#Team"]:

            print("En contra", row["Period"])
            arxs = dbs+"S0_%d.json" % int(row["Start Time [s]"])
            json.dump(red, open(arxs, 'w'))"""

            # RESETEO EN 2DO PERIODO ----------------------------
            if row["Period"] == 2 and flag:
                red = {}
                Links = []
                X, Y = [], []
                flag = 0


get_redes()
