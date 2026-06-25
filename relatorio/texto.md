# Aplicação de Visão Computacional na Transcrição de Notas e Acordes em Piano

**Autores:** Tiago Siqueira e William Kraus

---

## 1. Introdução

<!-- Contextualizar o problema: transcrição musical automática é um desafio em aberto,
     especialmente quando baseada em visão (em contraste com análise de áudio).
     Motivar o uso de visão computacional para detectar teclas pressionadas em piano. -->

### 1.1. Motivação

<!-- Por que usar visão computacional ao invés de (ou em complemento a) análise de áudio?
     Aplicações potenciais: ensino de piano, acessibilidade, geração de partituras, etc. -->

### 1.2. Escopo do Trabalho

<!-- Definir claramente o que o trabalho cobre e o que está fora de escopo.
     Ex.: foco em teclados eletrônicos, câmera fixa, sem reconhecimento de dinâmica (por enquanto). -->

---

## 2. Objetivos

### 2.1. Objetivo Geral

Desenvolver uma aplicação que utiliza técnicas de visão computacional para detectar teclas pressionadas de piano a partir da análise de imagens ou vídeo do instrumento.

### 2.2. Objetivos Específicos

1. **Identificar teclas pressionadas:** Detectar o momento em que determinadas teclas são pressionadas com base no posicionamento das mãos e em mudanças sutis de sombreamento das notas.
2. **Identificar a área do teclado:** Localizar a região do teclado na imagem e determinar a posição de cada tecla, com tolerância a variações de luminosidade e ângulo da câmera.
3. **Identificar intensidade:** Determinar a velocidade/intensidade com que uma nota é pressionada para inferir a dinâmica do som.

---

## 3. Fundamentação Teórica

<!-- Apresentar os conceitos teóricos utilizados no projeto. -->

### 3.1. Redes Neurais Convolucionais (CNNs)

<!-- Explicar o funcionamento de CNNs: camadas convolucionais, pooling, funções de ativação.
     Citar referências relevantes. -->

### 3.2. Classificação Multirrótulo (Multi-label Classification)

<!-- Explicar a diferença entre classificação multiclasse e multirrótulo.
     Justificar o uso de sigmoid independente por saída ao invés de softmax.
     Explicar a função de perda Binary Cross-Entropy (BCE). -->

### 3.3. Processamento Morfológico de Imagens

<!-- Explicar operações morfológicas: erosão, dilatação, abertura, fechamento.
     Explicar binarização adaptativa (adaptive threshold).
     Citar OpenCV como ferramenta. -->

### 3.4. Transformada de Hough

<!-- Explicar a transformada de Hough para detecção de linhas.
     Justificar seu uso na identificação de bordas horizontais do teclado. -->

### 3.5. Transformação de Perspectiva

<!-- Explicar a correção de perspectiva utilizada no recorte do teclado (warpPerspective).
     Importante para normalizar imagens capturadas em ângulo. -->

---

## 4. Metodologia

### 4.1. Cenário de Captura

<!-- Descrever o setup físico utilizado para gravação do dataset. -->

- **Câmera:** Logitech C270
- **Tripé:** 2,1 metros de altura
- **Instrumento:** Teclado Casio CTK-3500
- **Resolução de captura:** 640×480 (imagens recortadas para 640×128)

### 4.2. Pipeline Geral

<!-- Descrever o fluxo completo da aplicação, desde a captura do vídeo até a
     identificação das notas. Incluir um diagrama de alto nível. -->

```
Captura de Vídeo → Identificação da Área do Teclado → Recorte e Alinhamento
→ Pré-processamento → Rede Neural (CNN) → Notas Detectadas
```

### 4.3. Dataset

#### 4.3.1. Estrutura das Amostras

Cada amostra consiste em uma sequência de **três frames temporais** consecutivos. Cada frame é uma imagem em escala de cinza com dimensão **640×128** pixels, representando a região recortada do teclado.

<!-- Explicar a nomenclatura dos arquivos:
     {left_hand | right_hand}/{num_pressed_keys}/{fingers_index}/{Note}{Octave}_{frame}.png -->

#### 4.3.2. Recorte e Alinhamento

<!-- Descrever o processo de recorte da região do teclado a partir da imagem bruta.
     Mencionar suporte a recorte retangular e por perspectiva (skew).
     Referenciar o módulo crop.py e o config.json. -->

#### 4.3.3. Aumento de Dados (Data Augmentation)

As amostras passam por distorções controladas para aumentar a robustez do modelo:

- Variação de **luminosidade**
- Variação de **exposição**
- Variação de **contraste**
- Distorção de **perspectiva**

#### 4.3.4. Gravação do Dataset

<!-- Descrever o processo de gravação sincronizada com entrada MIDI.
     O listener MIDI (módulo midi.py) captura quais teclas estão pressionadas
     em cada momento, permitindo a geração automática dos rótulos. -->

#### 4.3.5. Balanceamento do Dataset

<!-- Explicar o mecanismo de capping das amostras sem nenhuma nota pressionada
     (cap_none) para evitar desbalanceamento. -->

#### 4.3.6. Divisão Treino/Teste

O dataset é dividido na proporção **80/20** (treino/validação). A divisão é estratificada para manter a distribuição de notas balanceada entre os dois conjuntos.

<!-- Descrever o algoritmo de split balanceado implementado em dataset.py
     (split_dataset_samples). -->

### 4.4. Modelo de Rede Neural

#### 4.4.1. Arquitetura

A rede neural é uma CNN com a seguinte arquitetura:

| Camada | Tipo | Entrada | Saída |
|--------|------|---------|-------|
| Entrada | — | (B, 3, 128, 640) | — |
| Bloco 1 - Conv2d | Conv2d(3, 16, kernel=3, padding=1) + ReLU | (B, 3, 128, 640) | (B, 16, 128, 640) |
| Bloco 1 - MaxPool | MaxPool2d(2, 2) | (B, 16, 128, 640) | (B, 16, 64, 320) |
| Bloco 2 - Conv2d | Conv2d(16, 32, kernel=3, padding=1) + ReLU | (B, 16, 64, 320) | (B, 32, 64, 320) |
| Bloco 2 - MaxPool | MaxPool2d(2, 2) | (B, 32, 64, 320) | (B, 32, 32, 160) |
| Flatten | Flatten | (B, 32, 32, 160) | (B, 163.840) |
| Camada Oculta | Linear(163.840, 512) + ReLU | (B, 163.840) | (B, 512) |
| Camada de Saída | Linear(512, 61) | (B, 512) | (B, 61) |

- A entrada consiste em **3 frames temporais** empilhados como canais (dimensão 3×128×640).
- A saída possui **61 neurônios**, um para cada tecla do teclado (C2 a C7).
- A ativação de saída utiliza **sigmoid independente** para cada neurônio, caracterizando uma classificação **multirrótulo**.

#### 4.4.2. Treinamento

- **Função de perda:** BCEWithLogitsLoss (Binary Cross-Entropy com logits), com pesos de classe calculados dinamicamente a partir da distribuição do dataset de treino.
- **Otimizador:** Adam (Adaptive Moment Estimation) com learning rate de 1×10⁻³.
- **Épocas:** Configurável (padrão: 20).
- **Batch size:** 32.
- **Critério de parada:** O treinamento pode ser interrompido ao atingir uma acurácia alvo (target accuracy).

<!-- Descrever detalhes adicionais: checkpoints automáticos, sistema de backups,
     hot-reloading do modelo durante testes. -->

#### 4.4.3. Métricas de Avaliação

<!-- Descrever como a acurácia é calculada (classificação exata de todas as 61 saídas
     binarizadas com threshold de 0.5).
     Discutir se outras métricas (F1, precision, recall por nota) foram ou devem ser avaliadas. -->

### 4.5. Identificação da Área do Teclado

A identificação automática da região do teclado na imagem segue o seguinte pipeline de processamento morfológico:

1. **Conversão para escala de cinza** e remoção de ruído (fastNlMeansDenoising).
2. **Binarização adaptativa** (Adaptive Threshold Gaussiano, block size=49, C=1) — tolerante a variações de iluminação.
3. **Abertura** com kernel 21×1 — remove ruído vertical e mantém componentes grandes.
4. **Fechamento** com kernel 1×7 — une brechas entre teclas brancas adjacentes.
5. **Detecção de linhas horizontais** (Transformada de Hough Probabilística) para scoring das componentes conexas.
6. **Seleção da componente conexa** com maior score (score = área × quantidade de linhas horizontais × proporção).
7. **Fechamento** com kernel 1×61 — preenche a área das teclas pretas.
8. **Abertura agressiva** com kernel 31×155 — retangulariza a área principal.
9. **Seleção final** da componente com maior área e proporção mínima 3:1.

<!-- Incluir imagens dos passos intermediários, se disponíveis (ignored_detection/). -->

#### 4.5.1. Problemas Atuais

<!-- Descrever os problemas identificados na apresentação:
     - Sensibilidade da binarização adaptativa a pequenas variações na imagem.
     - Seleção incorreta de componente conexa quando superfícies claras (ex.: chão)
       produzem componentes com proporção similar ao teclado.
     Hipóteses de melhoria:
     - Aprimorar o critério de seleção considerando: proporção 3:1, maior área,
       presença de linhas paralelas (Transformada de Hough). -->

### 4.6. Interface da Aplicação

<!-- Descrever as telas da aplicação (Kivy):
     - Tela de gravação do dataset (record.py)
     - Tela de recorte/calibração (crop.py)
     - Tela de teste/visão em tempo real (vision.py)
     Funcionalidades: freeze de imagem, ajuste de brilho/exposição/contraste,
     seleção de modelo, presets de vídeo, auto-crop, etc. -->

---

## 5. Resultados

<!-- Apresentar os resultados obtidos até o momento. -->

### 5.1. Dataset Construído

<!-- Descrever o tamanho do dataset, quantidade de amostras, distribuição de notas.
     Incluir gráficos ou tabelas com a distribuição. -->

### 5.2. Desempenho do Modelo

<!-- Apresentar métricas de acurácia, loss, e exemplos de predições.
     Incluir curvas de treinamento (loss por época), se disponíveis.
     Apresentar exemplos de acertos e erros. -->

### 5.3. Identificação do Teclado

<!-- Apresentar resultados qualitativos da detecção da área do teclado.
     Mostrar exemplos de sucesso e de falha. -->

### 5.4. Testes com Vídeo em Tempo Real

<!-- Descrever os testes manuais realizados com a aplicação rodando em tempo real.
     Incluir observações sobre latência, precisão e usabilidade. -->

---

## 6. Discussão

<!-- Analisar criticamente os resultados. -->

### 6.1. Limitações

<!-- Discutir limitações identificadas:
     - Sensibilidade a variações de iluminação e ângulo
     - Dependência do cenário de captura específico
     - Acurácia em acordes complexos
     - Identificação de intensidade (ainda não implementado)
     - Generalização para outros teclados -->

### 6.2. Trabalhos Futuros

<!-- Possíveis melhorias:
     - Redes mais profundas ou com arquiteturas mais sofisticadas (ResNet, etc.)
     - Uso de sequências temporais mais longas (LSTM, Transformer)
     - Detecção de intensidade/velocity
     - Dataset maior e mais diverso
     - Generalização para diferentes teclados e condições de iluminação
     - Saída em formato MIDI em tempo real -->

---

## 7. Conclusão

<!-- Resumir as contribuições do trabalho e os resultados alcançados.
     Reafirmar os objetivos atingidos e os que permanecem em aberto. -->

---

## Referências

<!-- Listar as referências bibliográficas utilizadas no formato ABNT ou IEEE. -->

<!-- Exemplo:
1. LECUN, Y. et al. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, v. 86, n. 11, p. 2278-2324, 1998.
2. OPENCV. OpenCV Documentation. Disponível em: https://docs.opencv.org/. Acesso em: ...
3. PYTORCH. PyTorch Documentation. Disponível em: https://pytorch.org/docs/. Acesso em: ...
-->
