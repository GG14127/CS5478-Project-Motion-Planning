# Motion Planning in Dynamic Environment
## CS5478 project code
Some resource comes from default project: https://github.com/NUS-LinS-Lab/Mobile-Manipulation  
### file sturcture
``` plaintext
Motion-Planning/  
├── resource/                   # resource for initializing environment, same as default project
│   ├── texture/               
│   └── urdf/                  
├── simulation/                 # Main directory for simulation modules  
│   ├── models/                 # Directory for model files including Policy network and Value network models  
│   ├── modules/                # Core algorithm modules  
│   │   ├── dwa.py              # Dynamic Window Approach (DWA) implementation  
│   │   ├── roboEnv.py          # RL environment module  
│   │   ├── RRTstar_nosmooth.py # RRT* algorithm without smoothing  
│   │   ├── RRTstar.py          # RRT* algorithm implementation  
│   │   └── simple_control.py   # Simple controller implementation  
│   ├── utils/                  # Utility files  
│   ├── init.py                 # pybullet initialization  
│   ├── rl_test.ipynb           # Reinforcement learning testing script  
│   ├── rl_train.ipynb          # Reinforcement learning training script  
│   ├── rrt_dwa.ipynb           # RRT combined with DWA in dynamic environment testscript  
│   └── rrt_static.ipynb        # RRT in static environment test script  
└── readme.md                   # Project description document  
```

### Test
To test RRT algorithm in static environment, run all `rrt_static.ipynb`.  
To test RRT algorithm conbined with DWA in dynamic environment, run all `rrt_dwa.ipynb`.  
To test RL model in static environment, run all `rl_test.ipynb`.  