def euler_step(x, t, v_func, dt):
    return x + v_func(x, t) * dt

def rk4_step(x, t, v_func, dt):
    k1 = v_func(x, t)
    k2 = v_func(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = v_func(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = v_func(x + dt * k3, t + dt)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def heun_step(x, t, v_func, dt):

    k1 = v_func(x, t)

    x_next_pred = x + k1 * dt
    k2 = v_func(x_next_pred, t + dt)

    return x + 0.5 * dt * (k1 + k2)