# Snake-Bot |  sb_5 - double q-learning

```bash
Input layer 
1: board state [4 x 12 x 12] (data_format = channels_first)

where the values in each pixel are:
    wall and snake’s body = 1
    freespace = 0
    snake’s head = 1
    food = 0.5
--------------------------------------------
               Hidden layers
--------------------------------------------
Output layer (linear regression - q values)
1: move left 
2: move up
3: move right
4: move down
```

The sample video of the actual gameplay is [here](https://youtu.be/CT7K99dArhA).

### Investigate the changes with TensorBoard

```bash
$ tensorboard --logdir files/training_logs
```