# Redes-Neurais-Artificiais-PP2.1

Aprendizado Supervisionado no Neurônio Perceptron

Neste projeto prático, o objetivo é implementar o algoritmo de treinamento mediante Aprendizado Supervisionado
do neurônio Perceptron de Rosenblatt aplicado em problemas de classificação. Nesta aplicação é utilizado um Jupyter
Notebook com o código-fonte desenvolvido na linguagem de Programação Python e fazendo
uso das bibliotecas numpy, random, math e matplotlib.

## Instalação

1. Crie o ambiente Conda a partir do arquivo `environment.yml`:
```bash
conda env create -f environment.yml
```

2. Ative o ambiente:
```bash
conda activate neural_net_env
```

3. Registre o kernel no Jupyter:
```bash
python -m ipykernel install --user --name neural_net_env --display-name "Redes Neurais"
```

4. Inicie o Jupyter Notebook ou Lab:
```bash
jupyter lab
```
ou
```bash
jupyter notebook
```

---

## To Do

2. Resolvendo um Problema Linearmente Separável
3. Experimentação
4. Validação Holdout em Problema Não-Linearmente Separável

### 1. Preparação do Ambiente
- [x] Obter o id e os dados que serão usados como entrada (`matriculas % 4`);
- [x] Estruturar a base do repositório no GitHub;
- [x] Preparar o ambiente de desenvolvimento com Anaconda e Jupyter com as bibliotecas necessárias (`numpy`, `matplotlib`, `scikit-learn`, `jupyter`);
- [x] Criar uma função para importar os dados binários;
- [ ] Criar os tipos básicos a serem trabalhados em código (estrutura de classes);
  - [ ] Classe `Perceptron`.

---

### 2. Resolvendo um Problema Linearmente Separável
- [ ] Utilizar o arquivo `dataAll.txt` como base;
- [ ] Implementar o algoritmo do Perceptron com as configurações:
  - Função de ativação degrau com limiar ϑ = 0
  - Taxa de aprendizado η = 0.1
  - Vetor de pesos inicializado com `U(-0.5, +0.5)`
- [ ] Treinar até convergência;
- [ ] Exibir na saída:
  - [ ] Vetor de pesos inicial
  - [ ] Número total de ajustes no vetor de pesos
  - [ ] Número de épocas até convergência
  - [ ] Gráfico com os exemplos plotados, reta separadora e cores por classe (vermelho = 0, azul = 1)

---

### 3. Experimentação
...

---

### 4. Validação Holdout em Problema Não-Linearmente Separável
...
