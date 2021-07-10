# CafeOrderNLP
GPT2 와 Bert 를 이용한 비대면 카페 주문 챗봇 만들기

##목적
nlp분야에서 가장 강력한 모델이라고 불리우는 bert와 gpt2를 이해하고, 다뤄 보기를 이번 프로젝트의 목표로 삼았습니다.<br>
##입력
학습할 때에는 Question과 Answer, intent가 labeling된 데이터를 입력으로 합니다. 학습이 완료된 모델을 사용할 때에는 Question을 넣으면 생성된 Answer가 출력으로 나옵니다. <br>
##관련 연구
인터넷을 찾아보다 pingpong팀에서 gpt2로 구현한 챗봇을 참고 많이 했습니다. 또한 SKT-AI에서 만든 kogpt2를 사용한 챗봇 또한 참고를 많이 했습니다. 두 프로젝트 모두 카페 대화 처럼 closed domain이 아니라 심리상담, 공감과 같이 domain이 딱히 정해지지 않았습니다. Text generation에 뛰어난 효과를 보이는 gpt-2를 활용하여 작업 하였습니다. 아래는 관련 링크입니다. <br>

- https://blog.pingpong.us/generation-model/ <br>
- https://github.com/haven-jeon/KoGPT2-chatbot <br>
##데이터
출처는 https://aihub.or.kr/aidata/85 에서 자료를 요청하여 받았습니다. 데이터에는 고객 (Question), 점원(Answer)의 대화가 있습니다. 또한 의도 또한 같이 labeling 되어 있습니다. 분류 모델에서는 training/validation/test 비율을 0.9/0/0.1로 하였고, 생성 모델에서는 0.7/0.2/0.1로 하 였습니다. 데이터 샘플은 아래와 같습니다. <br>
##사용한 플랫폼
구글 colab과 pytorch를 사용하였습니다. 그래프를 plotting 하는 데에는 tensorflow를 사용하였습니다. <br>

다음은 저희의 대략적인 architecture입니다. 학습 시킬때의 흐름도 입니다 <br>

Intent를 분류한 Bert 분류모델 입니다. <br>

Question과 intent에 따라 Answer을 만들어 내는 Gpt-2 모델입니다. <br>

##작업내용
위의 데이터를 학습할 수 있도록 전처리를 하였습니다. 그리고 git clone을 사용하여 오픈소스로 된 코드 위에 bert를 사용한 classifier를 구현을 하였습니다. 기존의 코드에서 학습할 때에는 intent를 구분 하지만 채팅할 때에 intent를 구분하지 않는다는 문제점을 발견하여 intent classifier를 구현하였습니다. train_torch.py에서 CharDataset과 KoGPT2Chat Class 를 제외한 나머지 부분은 아래의 링크를 참조하여 구현하였습니다. 또한 기존의 코드에는 KoGPT2Chat class에서 validation, test를 하지 않았지만 저희는 추가적으로 training_step과 training_dataloader를 변형하여validation_step과 validation_dataloader, test_step과 test_dataloader 을 구현하였으며, loss graph를 그려 봤습니다. <br>

Bert-classifier 참조 링크 : <br>
https://medium.com/@eyfydsyd97/bert%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C- classification-by-pytorch-2a6d4adaf162 <br>
kogpt2를 활용한 챗봇 링크 : https://github.com/haven-jeon/KoGPT2-chatbot <br>
