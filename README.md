# Pontos-Correlacao-de-Imagem

Este projeto implementa um comparador inteligente de imagens, capaz de identificar se duas imagens representam o mesmo local através de técnicas avançadas de visão computacional.

O sistema detecta características visuais, encontra correspondências entre elas e valida geometricamente esses matches usando **RANSAC**. Ao final, o programa gera uma imagem final mostrando os pontos equivalentes conectados por linhas e informa se as imagens são do *mesmo local* ou *locais diferentes*.

---

# Como Rodar o Projeto

Instale as dependências iniciando o temrinal no diretório do projeto, rode o comando:

```bash
pip install -r requirements.txt
```

# Adicione as imagens a serem comparadas na pasta Imagens_comparar

Images/
 ├── Imagens_comparar/
 │       imagem1.jpg
 │       imagem2.jpg
 └── Comparacao/

Por fim execute o comando abaixo para executar a aplicação:
```bash
python main.py
``` 
E o resultado será salvo automaticamente em: 

-> Images/Comparacao/Comparacao_resultado.jpg

