### Prerequisites

- Python 2.7
- tensorflow1.4.1

### Run
```
cd ./tests/
python daohang_pipei_10_risk90.py.py

91 #Enter destination when prompted with statement "请输入您的到达点"
0 #Enter the starting point when prompted with the statement "请输入您的出发点"

```






The output example after successful operation is as follows:

请输入您的到达点

您的到达点为91 

请输入您的出发点

您的出发点为0


******************* Finish generating one day order **********************
******************* Starting training Deep SARSA **********************
A_star  -----  A_star
搜索所用时间0.00206685066223s
从节点'0'到节点'91' 所需要的经过的节点是 [0, 10, 20, 30, 31, 41, 51, 61, 62, 72, 82, 81, 80, 90, 91] ,所需要的cost为 2944
Dijkstra  -----  Dijkstra
搜索所用时间0.000747919082642s
minimum steps:2944, path:[0, 10, 20, 30, 31, 41, 51, 61, 62, 72, 82, 81, 80, 90, 91]
