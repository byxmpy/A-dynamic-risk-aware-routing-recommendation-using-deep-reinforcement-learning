# A-dynamic-risk-aware-routing-recommendation-using-deep-reinforcement-learning
A dual-objective dynamic perception path planning method based on deep reinforcement learning is proposed, which perceives crime risk and path distance and generates dynamic optimal route recommendations.
### Prerequisites

- Python 2.7
- tensorflow1.4.1

### Data

The data is stored in npy files.If you want to use your own data, you can modify the two npy files at the code section below. The file should be a two-dimensional np matrix where the value at the Xth row and Yth column represents the distance or risk value from node X to node Y. 

```
origin_distance=np.load('90distance10.npy')
distance = origin_distance
origin_crime=np.load('90crime10.npy')
crime = origin_crime
```

Please note the following three points:
1. This code is designed for finding routes on a grid map, where each movement can only be made in the up, down, left, or right direction, and a single action cannot move to a non-adjacent grid. Therefore, distance values between non-adjacent nodes are invalid.
2. If a point in the distance matrix is 0, it means there is no connection between the corresponding two nodes.
3. The values in the matrix should be between 0 and 100. Code execution is not guaranteed if the values fall outside this range.

### Run
```
cd ./tests/
python daohang_pipei_10_risk90.py.py

91 #Enter destination when prompted with statement 
0 #Enter the starting point when prompted with the statement 

```


### Output

The output example after successful operation is in ```log.txt```

