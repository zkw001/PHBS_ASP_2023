# Garch Diffusion Model

## The new Milstein scheme from midterm exam

In a 2019 exam question, we derived the Milstein scheme for $v_t$ as
$v_{t+\Delta t} = v_t + \kappa(\theta - v_t)\Delta t + \omega v_t \sqrt{\Delta t}Z + \frac{\omega^2}{2}v_t(\Delta t)(Z^2 - 1)\Delta t$ for $Z \sim \mathcal{N}(0,1)$.

In the last equation of 2024 exam, we derived the new Milstein scheme as
$$v_{t+\Delta t} = \theta + (v_t - \theta)e^{-\kappa \Delta t} + \nu\sqrt{v_t} \left( \sqrt{\Delta t} Z + \frac{\nu}{2}(Z^2 - 1)\Delta t \right)$$

There are two Garch Diffusion Model in [Garch.py](https://github.com/PyFE/PyFENG/blob/main/pyfeng/garch.py) from Pyfeng, one is \textcolor(red){GarchUncorrBaroneAdesi2004} and the other is \textcolor(red){GarchMcTimeDisc}
