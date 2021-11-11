# DSSL-dynamic-sparse-subspace-learning
The code and data of DSSL model in my paper ***"Online Detection of Structural Change Points of High-dimensional Streamed Data by Dynamic Sparse Subspace Learning"*** are provided here. Any questions or discussions are welcome.

## The target of DSSL model
Detect the structural changes in a real-time manner

## Definition of structural changes
We refer to the change in linear relationship as a structural change, i.e., change of the manifold structure capturing the linear relationship among the high-dimensional streaming data. Figure 1 is a simple example with three time series illustrating the concept of structural change. In Figure 1(a), there are two change-points, i.e., $t=1000$ and $2000$. Once a change-point occurs, the linear relationship among these variables changes, e.g., from $Z=X-Y$ to $Z=4X+2Y$ at $t=1000$. Figure 1(b) shows the time series in a three-dimensional space. Clearly, each segment in Figure 1(a) corresponds to a plane or a linear manifold in Figure 1(b), and we can easily see the transition from one to another.  
![image](https://github.com/xuruiyu7/DSSL-dynamic-sparse-subspace-learning/blob/main/fig/subspace.pdf)
