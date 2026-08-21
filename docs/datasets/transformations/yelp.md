# Transformação do Yelp Open Dataset

## Fonte e objetivo

O Yelp Open Dataset utilizado contém 6.990.280 avaliações, 1.987.897 perfis de usuários e 150.346 estabelecimentos. O objetivo desta etapa foi converter os arquivos JSON Lines originais para os arquivos atômicos `.inter`, `.user` e `.item` do RecBole, preservando as avaliações explícitas e calculando apenas os atributos numéricos necessários para análises posteriores. Não foram realizados filtros de frequência, agrupamentos de conexão social ou tempo de permanência, nem divisão experimental dos dados.

## Procedimento de transformação

O arquivo de interações possui `user_id:token`, `item_id:token`, `rating:float` e `timestamp:float`. Cada avaliação foi mantida individualmente, sem agregação de pares repetidos. As datas originais foram interpretadas como UTC e convertidas para Unix timestamp. O maior instante observado, `2022-01-19 19:48:45`, foi adotado como data de referência do conjunto.

Antes da escrita das interações, os identificadores existentes em `user.json` e `business.json` foram indexados. Foram encontrados 32 identificadores de autores presentes em `review.json`, mas ausentes dos metadados de usuários. Esses autores eram responsáveis por 33 avaliações, removidas para impedir interações sem uma linha correspondente em `.user`. Nenhum estabelecimento referenciado estava ausente. O arquivo final contém 6.990.247 interações, uma redução de 0,00047%, e todos os usuários e estabelecimentos emitidos são efetivamente referenciados.

O arquivo `.user` contém exatamente `user_id`, `is_active`, `friend_count` e `tenure_years`. O nível de atividade foi calculado sobre `review_count` de todos os perfis originais. Pelo método de posto mais próximo, o percentil 95 foi 92 avaliações; usuários com `review_count >= 92` foram marcados como ativos, incluindo todos os empates no limiar. Isso produziu 100.370 usuários ativos (5,049%) e 1.887.527 inativos (94,951%). O valor original de `review_count` não foi exportado.

No arquivo desta versão, `friends` é uma sequência de identificadores separados por vírgula, enquanto `None` representa ausência de amigos. `friend_count` foi calculado contando esses identificadores e atribuindo zero a valores vazios ou `None`. O tempo de permanência foi calculado por

`tenure_years = max(0, data_referência − yelping_since) / 365,2425 dias`,

com arredondamento para seis casas decimais. Não foram gravados rótulos categóricos de conexão social ou tenure; sua discretização pertence à etapa analítica posterior.

O arquivo `.item` mantém identificador, nome, cidade, estado, CEP, coordenadas, média de estrelas, número de avaliações, situação de funcionamento e categorias. Espaços internos das categorias foram substituídos por `_`, permitindo que cada categoria composta seja um único token em uma sequência RecBole. Campos textuais tiveram espaços e quebras de linha normalizados.

## Estatísticas descritivas

| Característica | Resultado |
|---|---:|
| Usuários | 1.987.897 |
| Estabelecimentos | 150.346 |
| Interações finais | 6.990.247 |
| Período das avaliações | 16/02/2005 a 19/01/2022 |
| Média das avaliações | 3,749 estrelas |
| Densidade da matriz | 0,00234% |
| Esparsidade da matriz | 99,99766% |
| `review_count`, mediana / média / P95 / P99 | 5 / 23,39 / 92 / 311 |

| Distribuição relevante | Resultado |
|---|---|
| Notas | 1: 1.069.550; 2: 544.240; 3: 691.934; 4: 1.452.916; 5: 3.231.607 |
| Amigos | 44,19% com zero; mediana 2; média 52,93; P95 268; máximo 14.995 |
| Tenure | mediana 7,27 anos; média 7,29; P95 12,34; máximo 17,27 |
| Intervalos descritivos de tenure | <1 ano: 2,07%; 1–3: 7,43%; 3–5: 14,15%; 5+: 76,35% |
| Estabelecimentos | 119.698 abertos (79,61%); 30.648 fechados (20,39%) |
| Metadados ausentes de item | 73 CEPs e 103 conjuntos de categorias |

## Considerações metodológicas

A marcação de atividade usa a contagem declarada no perfil, não a quantidade de interações que permaneceu após a verificação referencial. A inclusão dos empates explica a proporção ligeiramente superior a 5%. Os intervalos de tenure apresentados na tabela são apenas descritivos e não existem como atributos no arquivo RecBole. De modo semelhante, `friend_count` permanece numérico para que os limites de conexão social possam ser definidos e avaliados posteriormente. Os resultados representam a base imediatamente após a conversão de formato, antes de filtros de densidade, particionamento dos dados ou construção dos grupos usados na análise de justiça.
