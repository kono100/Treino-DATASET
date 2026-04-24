Durante o treinamento do modelo YOLO, além dos hiperparâmetros configuráveis, existem métricas exibidas a cada época que indicam a evolução do aprendizado. Essas métricas são fundamentais para diagnosticar se o modelo está melhorando ou apresentando problemas.

Principais métricas do treinamento (log por época):

Epoch: indica o progresso do treinamento (ex: 12/15). Quanto mais épocas, maior a chance de aprendizado, desde que não haja overfitting.
GPU_mem: uso de memória (no seu caso 0G por estar em CPU). Não afeta a qualidade diretamente, mas limita o tamanho do modelo/lote.
box_loss: erro na localização das caixas (bounding boxes). Deve diminuir ao longo das épocas; valores altos indicam que o modelo não está aprendendo a posicionar corretamente os objetos.
cls_loss: erro na classificação das classes. Também deve diminuir; valores altos indicam confusão entre classes.
dfl_loss: refina a precisão das caixas (Distribution Focal Loss). Quanto menor, melhor o ajuste fino das detecções.
Instances: número de objetos presentes no batch atual. Varia conforme as imagens e não é um parâmetro ajustável direto.
Size: resolução usada no treino (ex: 640). Impacta diretamente precisão vs custo computacional.

Métricas de validação (qualidade do modelo):

Precisão (Box(P)): porcentagem de detecções corretas. Baixa precisão indica muitos falsos positivos.
Recall (R): capacidade de encontrar todos os objetos. Baixo recall indica que o modelo “não enxerga” vários objetos.
mAP50: média de precisão considerando IoU 0.5. Métrica geral de desempenho.
mAP50-95: métrica mais rigorosa (vários IoUs). É a principal para avaliar qualidade real do modelo.

Como corrigir e melhorar:

Se loss não diminui → aumentar épocas ou revisar dataset (labels erradas são comuns).
Se precisão baixa → reduzir falsos positivos ajustando conf ou melhorando dados negativos.
Se recall baixo → aumentar épocas ou melhorar variedade do dataset.
Se mAP baixo → geralmente falta mais dados ou melhor anotação.
Se métricas variam muito → dataset pequeno ou desbalanceado.

Resumo: métricas como box_loss, cls_loss e dfl_loss indicam o aprendizado interno, enquanto precisão, recall e mAP mostram a qualidade final; acompanhar a queda dos losses e o aumento do mAP é essencial para validar um bom treinamento.
