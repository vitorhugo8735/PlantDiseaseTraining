# PlantDiseaseTraining

Projeto de treinamento, avaliação e inferência de modelos de visão computacional para classificação de doenças em plantas. O checkpoint treinado pode ser consumido posteriormente pelo microserviço `PlantDiseaseAI`.

## Funcionalidades

- Treinamento com PyTorch e EfficientNet-B0.
- Validação automática da estrutura e dos rótulos do dataset.
- Avaliação com loss, acurácia, precisão, recall, F1-score e matriz de confusão.
- Predição de uma imagem ou de uma pasta inteira.
- Interface CLI com saída JSON para integração com outros serviços.
- Testes de fumaça para o carregamento do checkpoint e predições.

As classes padrão do modelo são:

1. `mildew`
2. `rose_P01`
3. `rose_R02`

> **Nota:** essas classes são usadas atualmente apenas nos testes iniciais do projeto. Elas não representam necessariamente o conjunto final de doenças ou culturas que será adotado em produção. Ao ampliar ou substituir as classes, atualize o `data.yaml`, os labels do dataset e a lista `DEFAULT_CLASS_NAMES` em `inference/predictor.py`, e treine um novo modelo.

## Requisitos

- Python 3.12 ou superior
- PyTorch e torchvision
- GPU CUDA opcional; o projeto usa CPU automaticamente quando CUDA não está disponível

## Instalação

Na raiz do projeto:

```bash
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Dataset

O dataset não é versionado no repositório. Por padrão, os scripts procuram primeiro por `dataset_redistributed/` e, caso esse diretório não exista, usam `dataset/`.

Cada dataset deve conter um `data.yaml` e os três splits abaixo:

```text
dataset_redistributed/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Cada imagem deve possuir um arquivo `.txt` correspondente em `labels/`. O arquivo `data.yaml` deve declarar as classes, por exemplo:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: mildew
  1: rose_P01
  2: rose_R02
```

Para redistribuir um dataset existente entre treino, validação e teste:

```bash
python tools/redistribute_dataset.py --dataset dataset
```

As proporções padrão são 70% para treino, 20% para validação e 10% para teste. Use `--seed`, `--train-ratio`, `--valid-ratio` e `--test-ratio` para alterá-las.

## Treinamento

O comando abaixo usa EfficientNet-B0, batch size 16, uma época e seleciona CUDA automaticamente quando disponível:

```bash
python train.py
```

Exemplo de treinamento personalizado:

```powershell
python train.py `
  --dataset-path dataset_redistributed `
  --epochs 20 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --device cuda
```

O treinamento salva:

- `checkpoints/best_model.pt`: melhor checkpoint de validação.
- `outputs/training_history.json`: histórico de loss e acurácia.

Esses diretórios são ignorados pelo Git.

## Avaliação

Para avaliar o checkpoint no split de validação:

```bash
python evaluate.py
```

Também é possível escolher o split e outros parâmetros:

```powershell
python evaluate.py `
  --dataset-path dataset_redistributed `
  --checkpoint-path checkpoints/best_model.pt `
  --split test `
  --batch-size 16 `
  --device cpu
```

Os splits aceitos são `train`, `valid` e `test`.

## Inferência

### Uma imagem

```bash
python predict.py --image caminho/para/imagem.jpg
```

Para usar outro checkpoint ou dispositivo:

```powershell
python predict.py `
  --image caminho/para/imagem.jpg `
  --checkpoint-path checkpoints/best_model.pt `
  --device cpu
```

### Lote de imagens

```powershell
python predict_batch.py `
  --input caminho/para/imagens `
  --output outputs/batch_predictions.csv `
  --batch-size 32
```

O CSV contém a classe prevista, a confiança e a probabilidade de cada classe. Imagens que não puderem ser processadas são reportadas no resumo da execução.

### Saída JSON para integração

`inference/inference_cli.py` retorna uma única resposta JSON no `stdout`, adequada para ser chamada por outro serviço:

```bash
python inference/inference_cli.py --image caminho/para/imagem.jpg
```

Exemplo de resposta:

```json
{"predictedClassId":0,"predictedClass":"mildew","confidence":0.9123,"probabilities":{"mildew":0.9123,"rose_P01":0.0521,"rose_R02":0.0356}}
```

## Testes

Execute os testes disponíveis com:

```bash
python -m unittest discover -s tests -v
```

O teste de fumaça requer um checkpoint em `checkpoints/best_model.pt` e imagens de exemplo em `dataset_redistributed/valid/images`. Sem esses artefatos, ele é ignorado.

## Estrutura do projeto

```text
.
├── inference/       # Preditor reutilizável e CLI JSON
├── src/
│   ├── configs/     # Configurações de dataset
│   ├── data/        # Dataset, transforms e validação
│   ├── evaluation/  # Métricas e avaliação
│   ├── models/      # EfficientNet e fábrica de modelos
│   └── training/    # Trainer, loss, optimizer e scheduler
├── tests/           # Testes automatizados
├── tools/           # Utilitários de preparação do dataset
├── train.py         # Entrada de treinamento
├── evaluate.py      # Entrada de avaliação
├── predict.py       # Predição de uma imagem
└── predict_batch.py # Predição em lote
```

## Observações

- O arquivo `export.py` está reservado para uma etapa futura de exportação do modelo.
- Datasets, checkpoints e saídas geradas não devem ser commitados.
