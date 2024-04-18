# Garch Diffusion Model

## The new Milstein scheme from midterm exam

In a 2019 exam question, we derived the Milstein scheme for $v_t$ as
$$v_{t+\Delta t} = v_t + \kappa(\theta - v_t)\Delta t + v v_t \sqrt{\Delta t}Z + \frac{v^2}{2}v_t(\Delta t)(Z^2 - 1)\Delta t$$ for $Z \sim \mathcal{N}(0,1)$.

In the last equation of 2024 exam, we derived the new Milstein scheme as
$$v_{t+\Delta t} = \theta + (v_t - \theta)e^{-\kappa \Delta t} + \nu\sqrt{v_t} \left( \sqrt{\Delta t} Z + \frac{\nu}{2}(Z^2 - 1)\Delta t \right)$$

There are two Garch Diffusion Model in [Garch.py](https://github.com/PyFE/PyFENG/blob/main/pyfeng/garch.py) from Pyfeng, one is **GarchUncorrBaroneAdesi2004** and the other is **GarchMcTimeDisc**

data from paper
first scheme
|  | 90 | 95 | 100 | 105 | 110 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 30 | 0.041937 | 0.427257 | 2.053600 | 5.491288 | 10.074733 |
| 60 | 0.241513 | 1.008496 | 2.896066 | 6.122492 | 10.354954 |
| 90 | 0.507320 | 1.522347 | 3.542143 | 6.673194 | 10.695507 |
| 120 | 0.790735 | 1.982263 | 4.087915 | 7.163421 | 11.044316 |
| 180 | 1.356611 | 2.789594 | 5.006922 | 8.020586 | 11.720208 |
| 252 | 2.002796 | 3.625134 | 5.928394 | 8.905051 | 12.474526 |
| 504 | 3.960264 | 5.932866 | 8.400366 | 11.341164 | 14.710294 |

second scheme
|  | 90 | 95 | 100 | 105 | 110 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 30 | 0.0421 | 0.4273 | 2.0534 | 5.4913 | 10.0749 |
| 60 | 0.2410 | 1.0083 | 2.8962 | 6.1223 | 10.3544 |
| 90 | 0.5057 | 1.5216 | 3.5419 | 6.6724 | 10.6938 |
| 120 | 0.7890 | 1.9821 | 4.0886 | 7.1633 | 11.0427 |
| 180 | 1.3552 | 2.7900 | 5.0082 | 8.0211 | 11.7190 |
| 252 | 2.0036 | 3.6277 | 5.9317 | 8.9078 | 12.4758 |
| 504 | 3.9612 | 5.9344 | 8.4021 | 11.3428 | 14.7114 |

diff
|  | 90 | 95 | 100 | 105 | 110 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 30 | 0.0002 | 0.0000 | -0.0002 | 0.0000 | 0.0002 |
| 60 | -0.0005 | -0.0002 | 0.0001 | -0.0002 | -0.0006 |
| 90 | -0.0016 | -0.0007 | -0.0002 | -0.0008 | -0.0017 |
| 120 | -0.0017 | -0.0002 | 0.0007 | -0.0001 | -0.0016 |
| 180 | -0.0014 | 0.0004 | 0.0013 | 0.0005 | -0.0012 |
| 252 | 0.0008 | 0.0026 | 0.0033 | 0.0027 | 0.0013 |
| 504 | 0.0009 | 0.0015 | 0.0017 | 0.0016 | 0.0011 |

## Time-discretization using exact mean and variance (Zhao 2009; Tubikanec et al. 2021)

$$dX_t = \lambda(\mu - X_t)\,dt + \sigma X_t\,dB_t$$
$$X_t = e^{-\left(\lambda + \frac{\sigma^2}{2}\right)t+\sigma B_t} \left( X_0 + \lambda \mu \int_0^t e^{\left(\lambda + \frac{\sigma^2}{2}\right)s-\sigma B_s} \, ds \right),$$

$$E(X_t|X_0) = X_0 e^{-\lambda t} + \mu (1 - e^{- \lambda t})$$

![Var(X_t|X_0)公式](https://github.com/zkw001/PHBS_ASP_2023/blob/main/CodeCogsEqn.png)


firstly, applying the expected value we obtained to the BSM model
|  | 90 | 95 | 100 | 105 | 110 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 30 | 0.0378 | 0.4274 | 2.0645 | 5.4921 | 10.0697 |
| 60 | 0.2331 | 1.0175 | 2.9193 | 6.1331 | 10.3477 |
| 90 | 0.5014 | 1.5407 | 3.5750 | 6.6936 | 10.6930 |
| 120 | 0.7900 | 2.0081 | 4.1276 | 7.1917 | 11.0485 |
| 180 | 1.3665 | 2.8251 | 5.0541 | 8.0589 | 11.7365 |
| 252 | 2.0217 | 3.6659 | 5.9785 | 8.9487 | 12.5002 |
| 504 | 3.9898 | 5.9745 | 8.4470 | 11.3853 | 14.7457 |
diff 
|  | 90 | 95 | 100 | 105 | 110 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 30 | -0.0041 | 0.0001 | 0.0109 | 0.0008 | -0.0050 |
| 60 | -0.0084 | 0.0090 | 0.0232 | 0.0106 | -0.0073 |
| 90 | -0.0059 | 0.0184 | 0.0329 | 0.0204 | -0.0025 |
| 120 | -0.0007 | 0.0258 | 0.0397 | 0.0283 | 0.0042 |
| 180 | 0.0099 | 0.0355 | 0.0472 | 0.0383 | 0.0163 |
| 252 | 0.0189 | 0.0408 | 0.0501 | 0.0436 | 0.0257 |
| 504 | 0.0295 | 0.0416 | 0.0466 | 0.0441 | 0.0354 |

secondly, using the log normal distribution to simulate the $\sigma_t$, and calculating the option price

step by step





## Approximate lV (Medvedev and Scaillet, 2007)
$$dv_t = \kappa(\theta - v_t)dt + \nu v_t dZ_t$$  
$$v_t = \sigma_t^2$$
the SDE for $\sigma_t$ is

$$d\sigma_t  = d\sqrt{v_t} = \frac{1}{2\sqrt{v_t}}dv_t - \frac{1}{8v_t\sqrt{v_t}}(dv_t)^2 $$

$$ = -\frac{\kappa}{2} \left(\frac{\theta}{\sigma_t} - \sigma_t\right) dt + \frac{\nu}{2} \sigma_t dZ_t - \frac{\nu^2}{8} \sigma_t dt $$

$$= \frac{1}{2} \left( \frac{\kappa\theta}{\sigma_t} - \left(\kappa + \frac{\nu^2}{4}\right) \sigma_t \right) dt + \frac{\nu}{2} \sigma_t dZ_t.$$

where $dZ_t = \rho dB_t + \sqrt{1-\rho^2}dW_t$
such that, 

$$ a(\sigma_t) = \frac{1}{2} \left( \frac{\kappa\theta}{\sigma_t} - \left(\kappa + \frac{\nu^2}{4}\right) \sigma_t \right)$$

$$ b(\sigma_t) = \frac{\nu}{2} \sigma_t$$

$$ b'(\sigma_t) = \frac{\nu}{2} $$ 

$$I_1(\theta; \sigma) = -\frac{\rho b \theta}{2},$$

$$I_2(\theta; \sigma) = \left( -\frac{5 \rho^2 b^2}{12 \sigma} + \frac{1}{6} \frac{b^2}{\sigma} + \frac{1}{6} \rho^2 bb' \right) \theta^2$$ 

$$+ \frac{a}{2} + \frac{\rho b \sigma}{4} + \frac{1}{24} \frac{\rho^2 b^2}{\sigma} + \frac{1}{12} b^2 - \frac{1}{6} \rho^2 bb',$$



