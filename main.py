import os
import cv2
from params import Tuning
from engine import ( abrir_e_tratar, gerar_descritores, comparar_descritores, aplicar_ransac, desenhar_relacoes)

DIR_IN = os.path.join("Images", "Imagens_comparar")
DIR_OUT = os.path.join("Images", "Comparacao")


def selecionar_arqs(pasta):
    validos = []
    suportados = (".jpg", ".jpeg", ".png")

    for arq in sorted(os.listdir(pasta)):
        if arq.lower().endswith(suportados):
            validos.append(os.path.join(pasta, arq))
        if len(validos) == 2:
            break

    if len(validos) < 2:
        raise FileNotFoundError("Adicione pelo menos duas imagens em Imagens/Comparar.")

    return validos[0], validos[1]


def analisar_pares(caminhoA, caminhoB):
    print("\n=== Comparação de Imagens ===\n")

    A = abrir_e_tratar(caminhoA)
    B = abrir_e_tratar(caminhoB)

    kpA, dsA = gerar_descritores(A)
    kpB, dsB = gerar_descritores(B)

    if dsA is None or dsB is None:
        raise RuntimeError("Não foi possível obter descritores válidos.")

    brutos, filtrados = comparar_descritores(dsA, dsB)

    inliers = aplicar_ransac(kpA, kpB, filtrados)

    print(f"Matches brutos:            {len(brutos)}")
    print(f"Após Ratio Test:          {len(filtrados)}")
    print(f"Inliers após RANSAC:      {len(inliers)}")

    saida = desenhar_relacoes(A, kpA, B, kpB, inliers)
    destino = os.path.join(DIR_OUT, "Comparacao_resultado.jpg")

    os.makedirs(DIR_OUT, exist_ok=True)
    cv2.imwrite(destino, saida)

    print("Resultado salvo em:", destino)

    if len(inliers) >= Tuning.INLIERS_MINIMUM:
        print("\n→ Interpretação: MESMO local.\n")
    else:
        print("\n→ Interpretação: locais diferentes.\n")


def main():
    arq1, arq2 = selecionar_arqs(DIR_IN)
    analisar_pares(arq1, arq2)


if __name__ == "__main__":
    main()
