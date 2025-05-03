# DQN_based_BTC_Trading_Algorithm
![image](https://github.com/user-attachments/assets/e2d87c65-7b70-4f7d-ac0d-d94e7a55514b)



## background


Most modern trading algorithms are dominated by CNNs and LSTMs.

LSTM is suitable for processing time series data. 
However, as CNNs became more adept at identifying patterns in charts, the two algorithms became dominant.
This algorithm aims to utilize the advantages of both models by processing time series using the POMDP definition of DQN and identifying patterns using CNN.
![image](https://github.com/user-attachments/assets/ad8caf9a-5c49-4e7c-bec9-e432e28843f7)


## Setup

Python 3.8 (for `dataclass` support) or higher is required.

pip install -r requirements.txt

or

```
pip install h5py
pip install keras
```

## Dataset

download from [HERE](https://drive.google.com/file/d/1H0koXr2UTDhCPQE_sJnZ3XPiZH07cXTn/view?usp=drive_link)
 (If it's impossible, contact us !)

## Result
### baseline
![image](https://github.com/user-attachments/assets/eb4233fe-2775-4918-9f04-b4441ee554cc)

### Ours
![image](https://github.com/user-attachments/assets/7d5a1abf-8b0a-4e82-a68b-483fb83470f6)
![image](https://github.com/user-attachments/assets/cc16586d-d210-41aa-87c3-211bbadddaeb)
