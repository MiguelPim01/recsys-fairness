# Transformação do conjunto de dados LastFM-360K

## Fonte e objetivo

O LastFM-360K contém perfis de usuários e registros de reprodução no formato `(usuário, artista, número de reproduções)`. A versão utilizada possui 17.559.530 registros de reprodução e 359.347 perfis. Nesta etapa, o objetivo foi apenas converter os arquivos TSV originais para os arquivos atômicos `.inter`, `.user` e `.item` aceitos pelo RecBole. Não foram aplicados filtros de densidade, divisão entre treino, validação e teste ou normalização das contagens para a escala de 1 a 5. Essas operações serão realizadas em uma etapa posterior, evitando misturar conversão de formato com decisões experimentais.

## Procedimento de transformação

Cada interação foi representada pelos campos `user_id:token`, `item_id:token` e `play_count:float`. Os perfis foram convertidos para `user_id`, gênero, idade, país e data de cadastro. Os itens contêm o identificador utilizado pelo modelo, o MusicBrainz Artist ID (MBID), quando disponível, e o nome do artista.

O MBID foi adotado como identificador preferencial do item. Quando ausente, foi criado um identificador com prefixo `name:` a partir do nome do artista. Para tornar esse fallback determinístico, o nome passou por normalização Unicode NFKC, conversão independente de maiúsculas e minúsculas, remoção de espaços nas extremidades e colapso de espaços consecutivos. Como consequência, grafias equivalentes são associadas ao mesmo item. Dos 268.602 itens finais, 160.153 (59,62%) possuem MBID e 108.449 (40,38%) utilizam o fallback nominal.

Foram aceitos apenas identificadores de usuário com 40 caracteres hexadecimais, correspondentes ao formato SHA-1 declarado pelo conjunto. O arquivo original continha dois identificadores inválidos — `dec 27, 2008` e `sep 20, 2008` — responsáveis por 86 linhas, que foram removidas. Também foram removidas uma interação com contagem zero e uma linha sem MBID ou nome de artista utilizável. Por fim, 421 linhas que resultavam no mesmo par usuário–item foram consolidadas pela soma das reproduções. Assim, a redução total foi de 509 linhas (0,0029%), resultando em 17.559.021 interações únicas por usuário e item.

Nas idades, valores vazios ou fora do intervalo de 1 a 120 anos foram escritos como ausentes. Essa regra invalidou 61 valores preenchidos, sem excluir seus usuários. Os demais metadados ausentes também foram preservados como células vazias.

## Estatísticas descritivas

| Característica | Resultado |
|---|---:|
| Usuários | 359.347 |
| Artistas | 268.602 |
| Interações finais | 17.559.021 |
| Total de reproduções | 3.778.545.583 |
| Densidade da matriz | 0,0182% |
| Esparsidade da matriz | 99,9818% |
| Reproduções por interação, mediana / média | 94 / 215,19 |
| Reproduções por interação, P95 / P99 / máximo | 750 / 1.880 / 419.157 |
| Interações por usuário, mediana / média / P95 | 49 / 48,86 / 63 |
| Interações por item, mediana / média / P95 | 2 / 65,37 / 140 |

A densidade foi calculada por `interações / (usuários × itens)`. A diferença entre a média e a mediana das reproduções, assim como o máximo elevado, evidencia uma distribuição de consumo fortemente assimétrica. A popularidade dos artistas também é concentrada: o percentil 99 corresponde a 1.093 usuários, enquanto o artista mais frequente aparece em 77.347 perfis.

| Metadado de usuário | Distribuição após a transformação |
|---|---|
| Gênero | masculino: 241.642 (67,24%); feminino: 84.930 (23,63%); ausente: 32.775 (9,12%) |
| Idade | 284.386 válidas; 74.961 ausentes (20,86%); mediana 23; média 25,36 |
| Países mais frequentes | Estados Unidos: 67.044; Alemanha: 31.651; Reino Unido: 29.902 |
| Período de cadastro | 29/10/2002 a 11/11/2009 |

## Considerações metodológicas

A transformação preserva a contagem bruta de reproduções. Portanto, `play_count` ainda não deve ser interpretado como avaliação explícita nem comparado diretamente às notas do Yelp. A futura normalização deverá ser definida e ajustada sem utilizar informações dos conjuntos de validação e teste. Da mesma forma, os valores demográficos ausentes permanecem distinguíveis no arquivo gerado e deverão receber tratamento explícito antes da construção dos agrupamentos de justiça. Os resultados acima descrevem exclusivamente a conversão para RecBole e constituem a linha de base anterior a qualquer amostragem ou filtragem experimental.
