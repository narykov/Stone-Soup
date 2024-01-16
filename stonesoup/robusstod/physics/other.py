import numpy as np

def get_noise_coefficients(GM):
    """ A function that returns adequate noise coefficients for Van Loan's method, ideally from physical considerations.
    We need these values that drive the process noise in the prediction step.
    """
    # ratio = 1e-2  # 2 orders are suggested by Simon following a Lee's manuscript
    ratio = 1e-14  # otherwise it `lands in Bolivia'
    # 1e-11 IDS;
    st_dev = ratio * GM
    q = st_dev ** 2
    q = 1
    q_xdot, q_ydot, q_zdot = q, q, q
    return np.array([q_xdot, q_ydot, q_zdot])
