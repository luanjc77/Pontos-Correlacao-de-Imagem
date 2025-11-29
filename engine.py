import cv2
import numpy as np
from params import Tuning

def abrir_e_tratar(caminho):
    grayscale = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise FileNotFoundError(f"Falha ao abrir o arquivo: {caminho}")

    ajustes = cv2.createCLAHE(
        clipLimit=Tuning.CLAHE_POWER,
        tileGridSize=(Tuning.CLAHE_CELLS, Tuning.CLAHE_CELLS)
    )

    tratada = ajustes.apply(grayscale.copy())
    return tratada


def gerar_descritores(imagem):
    extrator = cv2.SIFT_create()

    detectados = extrator.detect(imagem, None)  
    detectados, descritores = extrator.compute(imagem, detectados)  

    return detectados, descritores

def comparar_descritores(descA, descB):
    
    index_cfg = {"algorithm": 1, "trees": 5}
    search_cfg = {"checks": 70}

    fl = cv2.FlannBasedMatcher(index_cfg, search_cfg)

    bruto = fl.knnMatch(descA, descB, k=2)

    aprovados = []
    for pacote in bruto:
        if len(pacote) < 2:
            continue

        melhor, segundo = pacote
        if melhor.distance <= Tuning.DIST_RATIO * segundo.distance:
            aprovados.append(melhor)

    return bruto, aprovados


def aplicar_ransac(kpA, kpB, matches):

    if len(matches) < 4:
        return []

    origem = []
    destino = []

    for m in matches:
        origem.append(kpA[m.queryIdx].pt)
        destino.append(kpB[m.trainIdx].pt)

    origem = np.array(origem, dtype=np.float32)
    destino = np.array(destino, dtype=np.float32)

    _, mascara = cv2.findHomography(
        origem, destino,
        method=cv2.RANSAC,
        ransacReprojThreshold=Tuning.RANSAC_TOLERANCE
    )

    if mascara is None:
        return []

    inliers = []
    for match, valid in zip(matches, mascara.ravel()):
        if valid:
            inliers.append(match)

    return inliers


def padronizar_dimensoes(a, b):

    altura = max(a.shape[0], b.shape[0])

    if a.shape[0] < altura:
        dif = altura - a.shape[0]
        a = np.pad(a, ((0, dif), (0, 0)), mode="constant")

    if b.shape[0] < altura:
        dif = altura - b.shape[0]
        b = np.pad(b, ((0, dif), (0, 0)), mode="constant")

    return a, b


def desenhar_relacoes(imgA, kpA, imgB, kpB, matches):
    imgA, imgB = padronizar_dimensoes(imgA, imgB)

    combinado = np.hstack([imgA, imgB])
    combinado = cv2.cvtColor(combinado, cv2.COLOR_GRAY2BGR)

    desloc = imgA.shape[1]
    cor = (0, 255, 0)

    for m in matches:
        x1, y1 = map(int, kpA[m.queryIdx].pt)
        x2, y2 = map(int, kpB[m.trainIdx].pt)

        destino = (x2 + desloc, y2)

        cv2.circle(combinado, (x1, y1), 5, cor, -1)
        cv2.circle(combinado, destino, 5, cor, -1)
        cv2.line(combinado, (x1, y1), destino, cor, 2)

    return combinado
