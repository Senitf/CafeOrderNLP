# CafeOrderNLP
`GPT2` 와 `Bert` 를 이용한 비대면 카페 주문 챗봇 만들기

## 목적
자연어 처리 분야에서 가장 강력한 모델이라고 불리우는`Bert`와 `Gpt2`를 이해하고, 다뤄보는 것을 이번 프로젝트의 목표로 삼았습니다.<br>

## 입력
학습할 때에는 Question과 Answer, Intent가 Labeling된 데이터를 입력으로 합니다. 학습이 완료된 모델을 사용할 때에는 Question을 넣으면 생성된 Answer가 출력으로 나옵니다. <br>

## 관련 연구
인터넷을 찾아보다 pingpong팀에서 gpt2로 구현한 챗봇을 참고 많이 했습니다. 또한 SKT-AI에서 만든 kogpt2를 사용한 챗봇 또한 참고를 많이 했습니다. 두 프로젝트 모두 카페 대화 처럼 closed domain이 아니라 심리상담, 공감과 같이 domain이 딱히 정해지지 않았습니다. Text generation에 뛰어난 효과를 보이는 gpt-2를 활용하여 작업 하였습니다. 아래는 관련 링크입니다. <br>

- https://blog.pingpong.us/generation-model/ <br>
- https://github.com/haven-jeon/KoGPT2-chatbot <br>

## 데이터
출처는 https://aihub.or.kr/aidata/85 에서 자료를 요청하여 받았습니다. 데이터에는 고객 (Question), 점원(Answer)의 대화가 있습니다. 또한 의도 또한 같이 labeling 되어 있습니다. 분류 모델에서는 training/validation/test 비율을 0.9/0/0.1로 하였고, 생성 모델에서는 0.7/0.2/0.1로 하 였습니다. 데이터 샘플은 아래와 같습니다. <br>

## 사용한 플랫폼
- Google Colab.
- Pytorch
- Tensorflow

## 아키텍쳐

학습 과정에서의 흐름도 입니다 <br>

![정보검색_최종발표 002](https://user-images.githubusercontent.com/55660691/125150795-767d0b00-e17d-11eb-9560-c62d3b019e33.jpeg)
![정보검색_최종발표 003](https://user-images.githubusercontent.com/55660691/125150797-77ae3800-e17d-11eb-8899-658bfe55807f.jpeg)

Intent를 분류한 Bert 분류모델 입니다. <br>

![정보검색_최종발표 001](https://user-images.githubusercontent.com/55660691/125150790-711fc080-e17d-11eb-9e97-c6403efcb3d1.jpeg)

Question과 intent에 따라 Answer을 만들어 내는 Gpt-2 모델입니다. <br>

![정보검색_최종발표 001](https://user-images.githubusercontent.com/55660691/125150836-c5c33b80-e17d-11eb-8526-e37e0d8a8ee7.jpeg)

## 작업내용
위의 데이터를 학습할 수 있도록 전처리를 하였습니다. 그리고 git clone을 사용하여 오픈소스로 된 코드 위에 bert를 사용한 classifier를 구현을 하였습니다.<br>
기존의 코드에서 학습할 때에는 intent를 구분 하지만 채팅할 때에 intent를 구분하지 않는다는 문제점을 발견하여 intent classifier를 구현하였습니다.
<br>
train_torch.py에서 CharDataset과 KoGPT2Chat Class 를 제외한 나머지 부분은 아래의 링크를 참조하여 구현하였습니다.
<br>
또한 기존의 코드에는 KoGPT2Chat class에서 validation, test를 하지 않았지만 저희는 추가적으로 training_step과 training_dataloader를 변형하여 validation_step과 validation_dataloader, test_step과 test_dataloader 을 구현하였으며, loss graph를 그려 봤습니다.
<br>

## Bert-classifier 참조 링크
- https://medium.com/@eyfydsyd97/bert%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-classification-by-pytorch-2a6d4adaf162 <br>
## kogpt2를 활용한 챗봇 링크
- https://github.com/haven-jeon/KoGPT2-chatbot <br>

## 사용법

```
!pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install mxnet gluonnlp sentencepiece pandas transformers pytorch lightning
!pip install git+https://github.com/SKT-AI/KoGpt#egg=kogpt2
!git clone --recurse-submodules https://github.com/haven-jeon/KoGPT2-chatbot.git
!pip install tensorboardX
%cd KoGPT2-chatbot
!CUDA_VISIBLE_DEVICES=0 python train_torch.py --train --gpus 1 --max_epochs 30
%load ext tensorboard
%tensorboard --logdir=runs
!CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --chat
```

## 학습 과정

Question이 input으로 들어가면 intent가 output으로 나오는 Classifier의 학습 방법은 pre trained 된 bert 모델을 사용하여 fine tuning을 하였습니다. <br>
문장에 masking을 하여 bert의 특성인 양방향 encoding을 진행하도록 하였습니다. <br>
GPU를 사용하기 위해 병렬로 데이터를 넣어 주었으며 또한 문장의 맨 앞에 [CLS] 토큰을 삽입하여 classify를 하도록 하였습니다. <br>
classify를 진 행하고 결과 값에 손실함수는 classify 문제에서 많이 사용되는 cross entropy 를 사용하였습니다. <br>
Optimizer는 pytorch 에서 제공하는 AdamW를 사용하였습니다.
Question과 intent가 input으로 들어가면 Answer이 output으로 나오는 Generator(GPT-2)는 pytorch lightning module을 사용하였습니다. <br>
마찬가지로 병렬로 데이터를 처리할 수 있도록 하 였고, bert와 비슷하게 mask를 씌웠습니다. <br>
Bert와 다른 점은 Bert는 양방향인데에 반해 GPT-2 는 left to right로 다음 단어가 무엇이 나올지 예측하는 방식 이었습니다. <br>
GPT-2 모델도 pre trained된 모델이지만 한글로만 pre training을 시킨 모델 이어서 한글을 사용할 때에 더욱 효율 을 높이도록 하였습니다. <br>
모델을 사용하여 예측한 다음 단어와 원래 단어와의 cross entropy 한 값을 loss로 하였습니다. <br>
Optimizer는 pytorch에서 제공하는 AdamW를 사용하였습니다. <br>

## 결과 분석

Classifier의 성능은 test set에서 50퍼센트로 매우 낮게 나왔습니다. Epoch을 여러가지로 테스트 해봤지만 마찬가지로 50퍼센트 내외 였습니다. 데이터 부족 문제 인 것으로 추정 됩니다. 아래 의 그림은 classifier를 학습하면서 생성한 loss log 와 Generator의 loss log 입니다.

![정보검색_최종발표 010](https://user-images.githubusercontent.com/55660691/125150904-1f2b6a80-e17e-11eb-8670-fe2ab7b5ea3e.jpeg)

아래는 대화 예시입니다. 완벽하게 잘 되지는 않지만 그래도 어느정도 대화는 오가는 모습을 보 입니다. 출력 형식이 다듬어지지 않았습니다. ‘Customer > 점원 >’이 두 번 나오는데 두번째는 무시하면 됩니다.

![정보검색_최종발표 011](https://user-images.githubusercontent.com/55660691/125150907-2488b500-e17e-11eb-8da2-56867b5dcf71.jpeg)

## 미흡했던 것
- 대화의 흐름(문맥) 인식
단일 Input 에 대한 단일 답변은 어느정도 매끄러운 경우도 있었지만 대화를 지속했을 때, 그 흐름을 인식하는 것에는 어려움이 있었습니다. 해당 부분에 대해서는 `Multi Level Classification` 이나, 흐름에 대한 정보를 인식을 위한 `모델 자체에 대한 수정`이 필요하지 않을까 라고 생각했습니다.
- 특정 정보 인식 (음료나 메뉴의 가격 등)
상황에 맞는 답변만 고려하다보니 특정 메뉴의 가격을 묻거나, 성분을 묻는 경우에 그 메뉴와 전혀 상관없는 메뉴의 성분이나 가격을 답하고는 했습니다. 해당 부분에 대해서는 특정 가격인 성분에 관한 Train Data 를 `Text Augmentation` 등을 통해 강조하거나, 애초에 이 부분은 답변이 정해져있으므로, 문장을 따로 생성하지 않고 `준비되어있는 문장을 대신 출력`해도 좋을 것 같습니다.
