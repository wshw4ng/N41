* git clone 하고, 사용할 때 git pull하면 최신 버전으로 사용하기 편할 겁니다.


* Jupyter 노트북 환경에서 사용시, 아래와 같이 코드에 import하면 됩니다.
```python
from evaluation.eval_utils import *
```

* Jupyter 노트북 환경에서 사용시, import할 때 바뀐 코드를 반영하려면 아래 문구를 넣으면 됩니다.
```python
%load_ext autoreload
%autoreload 2
```

* 평가시, 아래 함수를 호출하면 됩니다.
    * attack_label은 실제 정답 라벨 (0:정상, 1:이상)을, predictions는 예측된 점수 (값의 범위 무관)를 넣으면 됩니다.
```python
evaluate_all(attack_label, predictions)
```
