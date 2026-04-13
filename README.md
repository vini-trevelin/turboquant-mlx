# Turboquant-MLX

## Atenção, produto interno e KV cache

No mecanismo de atenção dos transformers,

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

o termo $QK^\top$ representa a **matriz de compatibilidade** entre as queries atuais e as keys dos tokens já processados. Objetivamente, mede o quanto cada token atual deve “prestar atenção” nos tokens anteriores.

Recomputar continuamente essas keys e values torna-se rapidamente caro e ineficiente, utiliza-se **KV cache** (*key-value cache*), que armazena os tensores de keys e values já calculados ao longo da sequência, normalmente em uma estrutura do tipo:

`[batch, heads, seq_len, head_dim]`

Isso introduz um trade-off claro:

- **sem cache**: - memória, + recomputação;
- **com cache**: - recomputação, + consumo de memória.

O problema é que a dimensão `seq_len` cresce linearmente conforme novos tokens são adicionados ao contexto. Como keys e values são vetores de alta dimensão, o KV cache passa a consumir uma quantidade significativa de memória e se torna um dos principais gargalos da inferência em contexto longo - sendo o aumento deste um fator relevante para rápida melhora dos novos LLMs.

Objetivo central: **comprimir o KV cache minimizando a perda de informação relevante**. 

## Ponto crítico da compressão

O ponto crítico, porém, é que comprimir um vetor com pequeno erro de reconstrução não garante preservar bem os produtos internos entre queries e keys. 

Como esses produtos internos são a base do mecanismo de atenção ($QK^\top$), a qualidade da compressão não deve ser avaliada apenas por sua semelhança numérica com o vetor original (métricas como MSE), mas também por sua capacidade de manter a geometria relevante para o cálculo atencional (ex: vetor reconstruído com MSE maior pode ser mais relevante para attention se manter os alinhamentos com queries relevantes do que vetores com MSE menores) - isto é, manter o produto interno próximo do original.

TurboQuant não foca apenas em erro de reconstrução, mas também em **preservação de produto interno**, através de duas etapas

1. **um quantizador principal, focado em MSE**;
2. **um corretor residual, baseado em QJL de 1 bit**, para corrigir o impacto da compressão sobre o produto interno. 


### 1 — quantização principal orientada a MSE

Antes de quantizar, o vetor é transformado de forma aleatória para se tornar **mais amigável à quantização** (??). No paper do TurboQuant, isso aparece como uma **rotação aleatória** das entradas, deseja-se que as coordenadas do vetor resultante tenham distribuição mais regular, concentrada e quase independente em alta dimensão, para viabilizar quantizador escalar por coordenada com eficiência muito alta. 

#### Mas pq rotaçǎo aleatória??

Antes da rotação, a informação do vetor pode estar concentrada em poucas dimensões, o que torna a quantização por coordenada instável. Ex:

$$
x = [9.8,\ 0.1,\ 0.0,\ -0.2,\ 0.0,\ 0.1,\ 0.0,\ 0.0]
$$

Quase toda a informação está concentrada na primeira coordenada.


$$
Rx = [3.2,\ -2.8,\ 3.5,\ 2.9,\ -3.0,\ 3.1,\ -2.7,\ 2.6]
$$

A informação fica mais espalhada entre várias coordenadas. A rotação não cria informação nova nem “melhora” o vetor, apenas muda sua representação para uma base em que a quantização escalar funciona melhor.

O resultado é uma compressão com distorção próxima do melhor limite teórico conhecido para esse tipo de problema. O paper afirma que o TurboQuant fica dentro de um pequeno fator constante (`2.7x`) do limite inferior de taxa-distorção. 

Todavia, um quantizador ótimo para MSE não é necessariamente ótimo para atenção. O próprio paper destaca que quantizadores MSE-ótimos introduzem **viés na estimação de produto interno** previamente citado. Ou seja: mesmo que o vetor comprimido fique numericamente “parecido” com o original, ele ainda pode alterar sistematicamente os valores de $QK^\top$, que é a variável de interesse aqui.

### 2 — correção residual com 1-bit QJL

Depois da quantização principal, calcula-se o **resíduo**, aplica sobre esse resíduo uma versão de **1-bit QJL** (*Quantized Johnson–Lindenstrauss*). 

O QJL consiste em duas operações:

1. aplicar uma transformação do tipo **Johnson–Lindenstrauss (JL)**
2. quantizar o resultado para **apenas o bit de sinal**

Aqui tá o pulo do gato: utilidade disso não está em reconstruir perfeitamente o vetor residual, mas em melhorar a qualidade da **estimação de produto interno**, e, por consequência, a compressão dos meus vetores alinhados ao problema do KVcache/attention.


### PolarQuant

Mesma filosofia geral: usar uma transformação aleatória anterior à quantização para tornar os vetores mais estruturados e quantizáveis, evitando a necessidade de normalização explícita por bloco e, portanto, reduzindo overhead de memória. Neste caso, isso é feito via **random preconditioning + transformação polar** (deixo de ter $X$, $Y$ com bits de sinais relativos, restam apenas com $\rho$ e $\phi$). 

## Resultados Esperados

- **KV cache quantization**: qualidade praticamente neutra em `3.5 bits por canal` e degradação apenas marginal em `2.5 bits por canal`;  
- **nearest neighbor / vector search**: recall superior a métodos tradicionais de product quantization, com tempo de indexação praticamente nulo. 

> Como aumentar a capacidade prática de sistemas baseados em LLM e busca vetorial sem depender apenas de hardware maior, mais VRAM e mais custo?

## Motivação

Para estudo aplicado, minha motivação é menos “entender uma prova bonita” e mais avaliar se esse tipo de técnica pode ajudar a construir sistemas locais ou semi-locais mais viáveis para uso em produção. 

Se o gargalo de memória do KV cache realmente puder ser reduzido sem perda relevante de qualidade, isso abre espaço para teste em **LLMs locais com contexto maior**, **RAG sobre bases internas**, **busca semântica em documentos** e **agentes internos mais privados e mais baratos de operar**, utilizando-se da privacidade provida por modelos open-weights. 

Essa leitura é uma inferência operacional baseada nas propriedades que o paper reporta: compressão agressiva de KV cache, adequação a cenários online (sem retreinamento) e melhora prática em busca vetorial.


![alt text](docs/image.png)