# algorithms

`algorithms` folder is designed to contain implementation of algorithms or models. 
Content in `algorithms` can be loosely grouped components (e.g. models) or an algorithm has already has all
components chained together (e.g. Lightning Module, RL algo).
You should create a folder name after your own algorithm or baselines in it. 

An example can be found in `classifier` subfolder

The `common` subfolder is designed to contain general purpose classes that's useful for many projects, e.g MLP.

You should not run any `.py` file from algorithms folder. 
Instead, you write unit tests / debug python files in `debug` and launch script in `experiments`.

You are discouraged from putting visualization utilities in algorithms, as those should go to `utils` in project root.
