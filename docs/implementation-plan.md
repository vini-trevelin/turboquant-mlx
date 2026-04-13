# Plano de Implementação — TurboQuant em MLX (MacBook M4 Pro 24 GB)

## Objetivo

Implementar uma versão própria de **TurboQuant para KV cache** em **MLX**, com foco em código modular, legível e fácil de inspecionar, e comparar de forma controlada três regimes de inferência:

1. **sem KV cache**;
2. **com KV cache normal**;
3. **com KV cache + implementação de TurboQuant**.

A meta não é apenas reproduzir o paper conceitualmente, mas entender o método até o nível do código e medir, no meu hardware, o trade-off entre:

- uso de memória;
- velocidade;
- degradação de qualidade;
- comportamento em contexto longo.

---

## Escopo deste plano

Este plano assume algumas decisões desde o início:

- o backend inicial será **MLX**;
- o ambiente-alvo é **Apple Silicon**;
- o experimento principal será **KV cache quantization**, não vector search;
- o benchmark principal de qualidade será **Needle in a Haystack**;
- o projeto deve ser **o mais modular possível**, para que cada parte possa ser lida, testada e substituída sem confusão;

---

## Escolha inicial dos modelos

Vou começar com três modelos pequenos o bastante para rodar bem localmente e, ao mesmo tempo, diferentes o bastante para não virar um teste excessivamente específico de uma única família:

1. **mlx-community/Llama-3.2-1B-Instruct-4bit**
2. **mlx-community/Llama-3.2-3B-Instruct-4bit**
3. **mlx-community/Qwen3-4B-Instruct-2507-4bit**

### Por que esses três

- são modelos já empacotados para **MLX**;
- cabem com folga maior dentro do cenário de **24 GB**, o que é importante porque o experimento vai forçar contexto longo;
- permitem observar como o método se comporta em escalas diferentes;
- evitam começar grande demais e transformar o projeto em um problema de infraestrutura antes de ser um problema de pesquisa aplicada.

A ideia aqui é **não otimizar cedo demais**. Primeiro eu quero um conjunto pequeno, estável e repetível.

---

## Pergunta central do repositório

A pergunta prática do projeto será:

> até que ponto uma implementação própria de TurboQuant para KV cache, em MLX, consegue reduzir uso de memória e manter qualidade útil em contexto longo, quando comparada com inferência sem cache e com cache padrão?

---

## Comparações

### Regime A — sem KV cache

Serve como baseline conceitual mais puro. Mostra o custo de recomputar tudo e deixa evidente por que KV cache existe.

**Observação importante:** em contexto longo, essa baseline pode ficar impraticável muito rápido. Quando isso acontecer, o resultado “inviável” também será um resultado útil.

### Regime B — KV cache normal

É o baseline operacional realista. Mostra o comportamento padrão de inferência moderna e será a principal referência para medir:

- memória usada;
- latência;
- qualidade da saída;
- comportamento no benchmark de contexto longo.

### Regime C — KV cache com TurboQuant

É a implementação principal do projeto. A ideia é substituir o armazenamento do KV cache por uma versão comprimida baseada em:

- rotação aleatória fixa;
- quantização escalar por coordenada;
- correção residual inspirada em QJL.

O objetivo do regime C não é “ser bonito matematicamente”, mas sim competir de forma séria com o regime B em custo-benefício.

---

## Medições e métricas

### 1. Memória

- memória total do processo;
- crescimento da memória com o aumento do contexto;
- tamanho efetivo do KV cache em cada regime;
- compressão observada versus compressão teórica.

### 2. Velocidade

- tempo de prefill;
- tokens por segundo na geração;
- custo adicional de compressão e descompressão do cache;
- impacto líquido no throughput.

### 3. Qualidade

- acerto no benchmark **Needle in a Haystack**;
- estabilidade da resposta quando a “agulha” aparece em diferentes posições do contexto;
- eventual diferença de logit ou similaridade entre saídas do cache padrão e do cache comprimido.

### 4. Robustez

- se a implementação funciona para os três modelos sem mudanças ad hoc;
- se o comportamento degrada de forma previsível ao reduzir o bitrate;
- se a arquitetura modular permite trocar componentes sem quebrar o restante.

---

## Benchmark principal: Needle in a Haystack

O benchmark principal será um teste do tipo **Needle in a Haystack**.

- construir um contexto longo com muito texto irrelevante;
- inserir uma informação curta e específica em uma posição controlada do contexto;
- no final, fazer uma pergunta cuja resposta depende de recuperar exatamente essa informação.

### O que variar no benchmark

Quero variar pelo menos quatro coisas:

1. **comprimento do contexto**;
2. **posição da agulha** no contexto;
3. **modelo usado**;
4. **regime de cache**.

### Faixas iniciais sugeridas

- contextos curtos: 2k–4k tokens;
- contextos médios: 8k–16k tokens;
- contextos longos: 24k+ tokens, se o modelo e a memória permitirem.

### Observação - baseline sem cache

Para o regime **sem KV cache**, pode ser necessário limitar os comprimentos máximos ou o número de repetições, porque essa baseline tende a ficar inviável antes das outras. Isso não invalida o experimento; pelo contrário, ajuda a mostrar o ganho operacional do cache.

---

## Princípio de implementação: modularidade máxima

Como eu quero entender o código e não apenas fazer algo funcionar, a implementação deve ser separada em blocos pequenos, com interfaces claras.

### Módulos conceituais

#### 1. Camada de experimento
Responsável por:

- carregar modelo e tokenizer;
- selecionar regime de cache;
- rodar prefill e geração;
- coletar métricas;
- salvar resultados.

#### 2. Camada de benchmark
Responsável por:

- gerar instâncias de Needle in a Haystack;
- controlar tamanho do contexto;
- controlar posição da agulha;
- avaliar acerto.

#### 3. Camada de cache
Responsável por definir as três variantes:

- `NoKVCacheRunner`;
- `StandardKVCacheRunner`;
- `TurboQuantKVCacheRunner`.

Essa separação é importante porque a comparação central do projeto acontece exatamente aqui.

#### 4. Camada de compressão
Responsável pelos componentes internos do TurboQuant:

- rotação aleatória;
- quantizador escalar;
- bit-packing;
- descompressão;
- correção residual.

#### 5. Camada de medição
Responsável por:

- medir tempo;
- medir memória;
- agregar resultados;
- exportar tabelas e gráficos.

---

## Estratégia de implementação

## Fase 1 — baseline mínima funcionando

Objetivo: montar o circuito mínimo de benchmark com MLX antes de mexer em compressão.

### Entregáveis

- script único que carrega cada um dos 3 modelos;
- execução de geração simples com MLX;
- versão inicial do benchmark Needle in a Haystack;
- medição básica de latência e memória;
- comparação entre:
  - sem KV cache;
  - KV cache normal.

### Critério de saída da fase

Só avanço quando eu tiver uma tabela simples mostrando, para pelo menos um contexto e um modelo:

- tempo;
- memória;
- acerto.

Sem isso, qualquer implementação de TurboQuant corre o risco de nascer sem baseline confiável.

---

## Fase 2 — arquitetura do cache comprimido

Objetivo: implementar a infraestrutura do `TurboQuantKVCacheRunner` sem ainda buscar o melhor resultado final.

### Entregáveis

- estrutura de cache comprimido separada da lógica de benchmark;
- interface compatível com o fluxo do MLX usado no regime padrão;
- armazenamento comprimido por camada;
- mecanismo explícito de `compress()` e `decompress()`;
- testes unitários simples para garantir que o cache vai e volta sem quebrar forma, dtype e ordem dos dados.

### Decisão importante

Nesta fase, o foco é **legibilidade e rastreabilidade**, não micro-otimização. Eu quero conseguir abrir o código e seguir o caminho:

`KV original -> rotação -> quantização -> armazenamento -> reconstrução`

sem camadas desnecessárias de abstração.

---

## Fase 3 — implementação dos componentes do TurboQuant

Objetivo: implementar os blocos centrais do método em versões pequenas e auditáveis.

### Bloco A — rotação aleatória fixa

- gerar uma matriz ortogonal fixa por `head_dim`;
- garantir reprodutibilidade por seed;
- aplicar rotação e rotação inversa;
- validar preservação aproximada de norma e estabilidade numérica.

### Bloco B — quantização escalar por coordenada

- implementar um quantizador simples primeiro;
- depois evoluir para a variante inspirada no paper;
- suportar testes com diferentes bitrates.

### Bloco C — empacotamento binário

- guardar índices quantizados de forma compacta;
- permitir leitura clara do caminho de serialização e desserialização;
- manter implementação simples antes de tentar otimizações pesadas.

### Bloco D — correção residual inspirada em QJL

- implementar uma primeira versão separada do resto;
- medir impacto incremental dela sobre qualidade;
- só integrar definitivamente depois de medir a diferença contra a versão sem residual.

### Critério de saída da fase

Ter duas variantes funcionais do cache comprimido:

1. **rotação + quantização escalar**, sem residual;
2. **rotação + quantização escalar + residual**.

Isso é importante porque vai permitir medir o valor real da correção residual, em vez de apenas assumi-lo.

---

## Fase 4 — campanha principal de benchmarks

Objetivo: comparar de verdade os três regimes em todos os modelos escolhidos.

### Matriz principal de experimento

Para cada modelo:

- Regime A: sem KV cache;
- Regime B: KV cache normal;
- Regime C1: TurboQuant sem residual;
- Regime C2: TurboQuant com residual.

Para cada regime:

- múltiplos tamanhos de contexto;
- múltiplas posições da agulha;
- múltiplas repetições por configuração.

### Saídas desejadas

- tabela de memória por regime;
- tabela de latência por regime;
- taxa de acerto em Needle in a Haystack;
- curvas de degradação por tamanho de contexto;
- comparação direta entre TurboQuant com e sem residual.

### Perguntas que essa fase precisa responder

- quanto de memória eu realmente economizei?
- quanto de velocidade eu perdi ou ganhei?
- a qualidade caiu pouco ou caiu de forma inaceitável?
- a correção residual vale o custo extra?
- a resposta depende muito do modelo?

---

## Métricas mínimas que eu quero salvar em cada execução

Cada execução deve salvar pelo menos:

- nome do modelo;
- regime de cache;
- tamanho do contexto;
- posição da agulha;
- bitrate usado;
- tempo de prefill;
- tempo de geração;
- tokens por segundo;
- pico de memória;
- tamanho estimado do KV cache;
- resposta gerada;
- acerto/erro no benchmark.

Sem isso, fica muito difícil comparar versões do projeto com honestidade.

---

## Critérios de sucesso do projeto

Vou considerar o projeto bem-sucedido se, ao final da primeira grande rodada de testes, eu tiver:

1. uma implementação própria, legível e modular de TurboQuant para KV cache em MLX;
2. comparação clara entre **sem cache**, **cache padrão** e **cache comprimido**;
3. evidência concreta de trade-off entre memória, velocidade e qualidade;
4. um benchmark reproduzível em Needle in a Haystack;
5. entendimento real do que no método traz valor e do que é apenas detalhe teórico elegante.

---

## Riscos práticos já assumidos

### 1. A baseline sem cache pode ficar inviável cedo demais

Isso é esperado. A forma correta de lidar com isso não é esconder o resultado, mas registrar até onde ela ainda é comparável.

### 2. O residual pode complicar bastante a implementação

Por isso ele entra como módulo separado. Se ele atrasar o projeto, eu ainda consigo obter um resultado útil com a variante sem residual.

### 3. O ganho de memória pode não se traduzir automaticamente em ganho de throughput

Compressão e descompressão têm custo. O projeto precisa medir isso, não assumir.

### 4. O comportamento pode variar bastante por modelo

Isso não é problema; é, na verdade, uma parte importante do que eu quero descobrir.

---

## Primeiros passos concretos

1. validar ambiente MLX e `mlx-lm` no M4 Pro;
2. rodar os 3 modelos escolhidos em geração simples;
3. implementar benchmark mínimo de Needle in a Haystack;
4. medir baseline de **sem cache** e **cache normal**;
5. implementar estrutura vazia do `TurboQuantKVCacheRunner`;
6. integrar rotação aleatória;
7. integrar quantização escalar simples;
8. medir primeira versão comprimida, mesmo que ainda imperfeita;
9. só depois adicionar residual inspirado em QJL.

---