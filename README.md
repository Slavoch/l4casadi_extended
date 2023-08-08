## Extended l4casadi for weights handling
### Description
This project makes you capable to cower the following issue:
You have to create a casadi version of the torch model once and have access to the weights, without reconstruction of the casadi model.
### Usage
To use this stuff you have to make your torch. Modul by inheriting the class ``Cs_Torch``
```
class MultiLayerPerceptron(Cs_Torch):
    pass
```
And fill with everything you want, with the list of exceptions:
 - use ``self.linear`` instead of ``nn.Linear``
 - call ``self.post_init()`` at the end of ``__init__``
 - dont do ``nn.ModuleList``
 - forward have to start with 
    ```
    def forward(self, inp):
        x, self.weights_package = self.pack_up_input(inp)
    ```
 - input vector should be a row  or pass it at the beginning as ``x = self.input_layer(x.T)``
 
 After model creation, you can call it in 2 different ways
 - Torch
    ```
    x = np.random.random(pyTorch_model.input_size)
    x_torch = torch.tensor(x, dtype=torch.float32)
    pyTorch_model.to_torch()
    pyTorch_model(x_torch).detach().numpy()
    ```
 - Casadi
    ```
    x_sym = cs.MX.sym('x', pyTorch_model.input_size, 1)
    w_sym = cs.MX.sym('w', pyTorch_model.weights_shape, 1)
    inp_sym = cs.vcat([x_sym, w_sym])

    f_sym = pyTorch_model.to_casadi()(x_sym, w_sym)
    cs_model_with_w = cs.Function('y', [inp_sym], [f_sym])
    x_cs = cs.DM(x)
    random_w = cs.DM(np.random.random(pyTorch_model.weights_shape))
    x_with_w = cs.vcat([x_cs, random_w])
    cs.evalf(cs_model_with_w(x_with_w))
    ```
    The most expensive operation is ``pyTorch_model.to_casadi()(x_sym, w_sym)``, you have to do it once.


