The project is based on hobbies and hands-on production.
It aims to identify defects in the injection molding industry that are not satisfied with injection molding. The product is perfusion wind blade.
1. The cross-flow fan blade is shown in the figure below. Its detection is mainly to detect the injection molding of the leaf. Therefore, the premise of solving this task is to identify whether each leaf is OK. The method of judging leaf NGorOK is temporarily classified as a binary classification task.

![image](img\1675154455578.jpg)

2. This task does not require deep learning to complete. The random forest classifier can be used to solve the leaf classification task, and opencv-python is used for preprocessing.
The Baidu cloud disk link of the data set is as follows, just unzip it and put it in the root directory.
https://pan.baidu.com/s/1tJaU0aZzPJC23HIi-HkhJw?pwd=sw8h 
![image](img\1675154586066.jpg)

3. python main.py -img_path 'img_path'


4.make ng img place 
loaddatas.loaddata.get_ng()
Then manually identify NG leaves and put them into ng_train