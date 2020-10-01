# Lab 4 - Nonlinear MAP estimation
Welcome to Lab 4!
We will here experiment with nonlinear MAP estimation.

I have made the following examples:
  - [ex1-nonlinear-least-squares-estimation.py](ex1-nonlinear-least-squares-estimation.py)
  - [ex2-map-estimation.py](ex2-map-estimation.py)

They both use the Gauss-Newton algorithm implemented in [optim.py](optim.py)
I have this week chosen to give a number of suggested experiments based on these examples.

You can install all dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Suggested experiments
1. Play around with [ex1-nonlinear-least-squares-estimation.py](ex1-nonlinear-least-squares-estimation.py)
    - How does it perform on initial states far from the true state?
    - What happens when you change the true pose?
    - Does the cost always decrease?
    - Can you make the problem diverge?
    
2. Implement Levenberg-Marquardt in [optim.py](optim.py)
     - Use this method instead of Gauss-Newton, and compare.
    
3.  Play around with [ex2-map-estimation.py](ex2-map-estimation.py)
     - Try changing the geometry and measurement noise and see how this affects the covariance estimate.
     - What happens when some measurements are more noisy than others?
     - Use Levenberg-Marquardt, how does this compare?
     - Add a prior distribution on the pose, and see how this affects the results.
    
 4. Extra: [Iteratively Closest Point (ICP)](https://www.youtube.com/watch?v=djnd502836w)
     - Simulate or download a point cloud you can use for ICP.
       Let one point cloud have the role as "world points", and the other "observed points" as in the examples.
       You can for example split a point cloud into two different sets by drawing points randomly
       from the original point cloud, and then transform the "observed point cloud" as in the examples. 
     - Keep the estimation framework from before (you don't need to implement the estimation methods in the video),
       but update the model so that it determines point correspondences using an ICP strategy.
     - Estimate pose using ICP!
