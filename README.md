# Symbolic Neural Architecture Search for Differential Equations

To reproduce experiments in the paper:
```sh
# Malthus model
python run.py --experiment 1

# Burgers' equation
python run.py --experiment 2

# Euler-Tricomi equation
python run.py --experiment 3
```

`eq1.ipynb`, `eq2.ipynb`, `eq3.ipynb` contain notebooks using the architecture.
`eq1.ipynb` has a more in-depth explanation of how to use this code.

## Requirements
```
sympy >= 1.12
matplotlib >= 3
jax >= 0.4.9
jaxlib >= 0.4.9
jaxtyping >= 0.2.16
optax >= 0.1.5
```

Run `pip install -r requirements.txt` to install.