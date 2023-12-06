import numpy as np
import progressbar


# Cash Karp parameters: http://www.aip.de/groups/soe/local/numres/bookcpdf/c16-2.pdf
a2,a3,a4,a5,a6 = 1./5., 3./10., 3./5., 1., 7./8.
b21 = 1./5.
b31,b32 = 3./40.,9./40.
b41,b42,b43 = 3./10, -9./10, 6./5.
b51,b52,b53,b54 = -11./54., 5./2., -70./27., 35./27.
b61,b62,b63,b64,b65 = 1631./55296., 175./512., 575./ 13824., 44275./110592., 253./4096.
c1,c2,c3,c4,c5,c6 = 37./378., 0., 250./621., 125./594., 0., 512./1771.
cc1,cc2,cc3,cc4,cc5,cc6 = 2825./27648., 0., 18575./48384., 13525./55296., 277/14336., 1./4.

def _rkf45_stepper(f, t, y, h):
    k1 = h * f(t,y)
    k2 = h * f(t + a2 * h, y + b21 * k1)
    k3 = h * f(t + a3 * h, y + b31 * k1 + b32 * k2)
    k4 = h * f(t + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3)
    k5 = h * f(t + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
    k6 = h * f(t + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
    nexty_5 = y + c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 +c6*k6
    nexty_4 = y + cc1*k1 + cc2*k2 + cc3*k3 + cc4*k4 + cc5*k5 +cc6*k6
    delta = np.linalg.norm(nexty_5 - nexty_4)
    return nexty_5, delta

def _rkf45_stepper_gdual(f, t, y, h):
    k1 = h * f(t,y)
    k2 = h * f(t + a2 * h, y + b21 * k1)
    k3 = h * f(t + a3 * h, y + b31 * k1 + b32 * k2)
    k4 = h * f(t + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3)
    k5 = h * f(t + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
    k6 = h * f(t + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
    nexty_5 = y + c1*k1 + c3*k3 + c4*k4 + c6*k6
    return nexty_5

def rkf45(f, t0, y0, tf, tol=1e-8, h0=1e-3):
    """Runge-Kutta-Fahlberg adaptive step-size integrator (floats)

    Args:
        f (``function``): a function with signature f(t, y) returning a numpy array with the r.h.s. of the differential equations to numerically integrate.
        t0 (``float``): starting time
        y0 (``array like``): initial conditions
        tf (``float``): final time
        tol (``float``): absolute tolerance to maintain
        h0 (``float``): initial step to take

    Returns:
        numpy.array, numpy.array: an array containing the times and a second array containing the state at the times.
    """
    retvalt = [t0]
    retvals = [y0]
    current_t = 0.

    while (current_t + h0 < tf):
        # Call the stepper
        nexty, delta = _rkf45_stepper(f, current_t, retvals[-1], h0)
        # Store the results
        retvals.append(nexty)
        current_t = retvalt[-1] + h0
        retvalt.append(current_t)
        # Update step-size
        if delta > 0:
            h0 = (tol / delta) ** (0.25) * 0.84 * h0
    # Make the last step
    h0 = tf - current_t
    nexty, delta = _rkf45_stepper(f, current_t, retvals[-1], h0)
    retvals.append(nexty)
    current_t = retvalt[-1] + h0
    retvalt.append(current_t)
    return retvalt, np.array(retvals)


def rkf45_gduals(f, t0, y0, tf, tol=1e-6, h0=1e-3):
    print("Computing the stepsize schedule ... ", flush=True)
    y0float = [it.constant_cf for it in y0]
    t,_ = rkf45(f, t0, y0float, tf, tol, h0)
    h_list = np.diff(t)
    retval = [y0]
    print("Building the Taylor Map: " + str(len(h_list)) + " steps needed" , flush=True)
    for h in progressbar.progressbar(h_list):
        retval.append(_rkf45_stepper_gdual(f, t0, retval[-1], h))
    return t, np.array(retval, dtype=object)

