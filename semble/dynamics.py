import numpy as np
from . import initial_state


class Dynamics:

    def __init__(self, state_dim, control_dim, mask=None,input_mask=None):
        self.n = state_dim
        self.m = control_dim

        # self.mask = mask if mask is not None else self.n * (1, )
        self.mask = mask if mask is not None else np.ones(self.n)
        self.p = sum(self.mask)
        self.input_mask = None
        self.locations = None

        self._method = "RK45"

    def __call__(self, x, u):
        return self._dx(x, u)

    def default_initial_state(self) -> initial_state.InitialStateGenerator:
        """Returns an instance of the default initial state sampler."""
        return initial_state.GaussianInitialState(self.n)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def default_method(self) -> str:
        return self._method

    def dims(self):
        return (self.n, self.m, self.p)


class LinearSys(Dynamics):

    def __init__(self, a, b):
        a = np.array(a)
        b = np.array(b)

        super().__init__(a.shape[0], b.shape[1])

        self.a = a
        self.b = b

    def _dx(self, x, u):
        return self.a @ x + self.b @ u


class VanDerPol(Dynamics):

    def __init__(self, damping):
        super().__init__(2, 1)

        self.damping = damping

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -p + self.damping * (1 - p**2) * v + u.item()

        return (dp, dv)


class FitzHughNagumo(Dynamics):

    def __init__(self, tau, a, b):
        super().__init__(2, 1)
        self._method = "BDF"

        self.tau = tau
        self.a = a
        self.b = b

    def _dx(self, x, u):
        v, w = x

        dv = 50 * (v - v**3 - w + u.item())
        dw = (v - self.a - self.b * w) / self.tau

        return (dv, dw)


class Pendulum(Dynamics):

    def __init__(self, damping, freq=2 * np.pi):
        super().__init__(2, 1)
        self.damping = damping
        self.freq2 = freq**2

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -self.freq2 * np.sin(p) - self.damping * v + u.item()

        return (dp, dv)


class HodgkinHuxleyFS(Dynamics):

    def __init__(self):
        super().__init__(4, 1, (1, 0, 0, 0))
        self._method = "BDF"

        self.time_scale = 100.
        self.v_scale = 100.

        # Parameters follow
        #   A. G. Giannari and A. Astolfi, ‘Model design for networks of
        #   heterogeneous Hodgkin–Huxley neurons’,
        #   Neurocomputing, vol. 496, pp. 147–157, Jul. 2022,
        #   doi: 10.1016/j.neucom.2022.04.115.
        self.c_m = 0.5
        self.v_k = -90.
        self.v_na = 50.
        self.v_l = -70.
        self.v_t = -56.2
        self.g_k = 10.
        self.g_na = 56.
        self.g_l = 1.5e-2

    def default_initial_state(self):
        return initial_state.HHFSInitialState()

    def _dx(self, x, u):
        v, n, m, h = x

        # denormalise first state variable
        v *= self.v_scale

        dv = (u.item() - self.g_k * n**4 *
              (v - self.v_k) - self.g_na * m**3 * h *
              (v - self.v_na) - self.g_l * (v - self.v_l)) / (100. * self.c_m)

        a_n = -0.032 * (v - self.v_t -
                        15.) / (np.exp(-(v - self.v_t - 15.) / 5.) - 1)
        b_n = 0.5 * np.exp(-(v - self.v_t - 10.) / 40.)
        dn = a_n * (1. - n) - b_n * n

        a_m = -0.32 * (v - self.v_t -
                       13.) / (np.exp(-(v - self.v_t - 13.) / 4.) - 1)
        b_m = 0.28 * (v - self.v_t - 40.) / (np.exp(
            (v - self.v_t - 40.) / 5.) - 1)
        dm = a_m * (1. - m) - b_m * m

        a_h = 0.128 * np.exp(-(v - self.v_t - 17.) / 18.)
        b_h = 4. / (1. + np.exp(-(v - self.v_t - 40.) / 5.))
        dh = a_h * (1. - h) - b_h * h

        return tuple(self.time_scale * dx for dx in (dv, dn, dm, dh))


class HodgkinHuxleyRSA(Dynamics):

    def __init__(self):
        super().__init__(5, 1, (1, 0, 0, 0, 0))
        self._method = "BDF"

        self.time_scale = 100.
        self.v_scale = 100.

        # Parameters follow
        #   A. G. Giannari and A. Astolfi, ‘Model design for networks of
        #   heterogeneous Hodgkin–Huxley neurons’,
        #   Neurocomputing, vol. 496, pp. 147–157, Jul. 2022,
        #   doi: 10.1016/j.neucom.2022.04.115.
        self.c_m = 1.0
        self.v_k = -90.
        self.v_na = 56.
        self.v_l = -70.3
        self.v_t = -56.2
        self.g_k = 6.
        self.g_m = 0.075
        self.g_na = 56.
        self.g_l = 2.05e-2
        self.t_max = 608.

    def default_initial_state(self):
        return initial_state.HHRSAInitialState()

    def _dx(self, x, u):
        v, p, n, m, h = x

        # denormalise first state variable
        v *= self.v_scale

        dv = (u.item() - (self.g_k * n**4 + self.g_m * p) *
              (v - self.v_k) - self.g_na * m**3 * h *
              (v - self.v_na) - self.g_l * (v - self.v_l)) / (100. * self.c_m)

        t_p = self.t_max / (3.3 * np.exp(
            (v + 35.) / 20.) + np.exp(-(v + 35.) / 20.))

        dp = (1. / (1 + np.exp(-(v + 35) / 10.)) - p) / t_p

        a_n = -0.032 * (v - self.v_t -
                        15.) / (np.exp(-(v - self.v_t - 15.) / 5.) - 1)
        b_n = 0.5 * np.exp(-(v - self.v_t - 10.) / 40.)
        dn = a_n * (1. - n) - b_n * n

        a_m = -0.32 * (v - self.v_t -
                       13.) / (np.exp(-(v - self.v_t - 13.) / 4.) - 1)
        b_m = 0.28 * (v - self.v_t - 40.) / (np.exp(
            (v - self.v_t - 40.) / 5.) - 1)
        dm = a_m * (1. - m) - b_m * m

        a_h = 0.128 * np.exp(-(v - self.v_t - 17.) / 18.)
        b_h = 4. / (1. + np.exp(-(v - self.v_t - 40.) / 5.))
        dh = a_h * (1. - h) - b_h * h

        return tuple(self.time_scale * dx for dx in (dv, dp, dn, dm, dh))


class HodgkinHuxleyIB(Dynamics):

    def __init__(self):
        super().__init__(7, 1, (1, 0, 0, 0, 0, 0, 0))
        self._method = "BDF"

        self.time_scale = 100.
        self.v_scale = 100.

        # Parameters follow
        #   A. G. Giannari and A. Astolfi, ‘Model design for networks of
        #   heterogeneous Hodgkin–Huxley neurons’,
        #   Neurocomputing, vol. 496, pp. 147–157, Jul. 2022,
        #   doi: 10.1016/j.neucom.2022.04.115.
        self.c_m = 1.0
        self.v_k = -90.
        self.v_ca = 120.
        self.v_na = 56.
        self.v_l = -70
        self.v_t = -56.2
        self.g_k = 5.
        self.g_m = 0.03
        self.g_ca = 0.2
        self.g_na = 50.
        self.g_l = 0.01
        self.t_max = 608.

    def default_initial_state(self):
        return initial_state.HHIBInitialState()

    def _dx(self, x, u):
        v, p, q, s, n, m, h = x

        # denormalise first state variable
        v *= self.v_scale

        dv = (u.item() - (self.g_k * n**4 + self.g_m * p) *
              (v - self.v_k) - self.g_ca * q**2 * s *
              (v - self.v_ca) - self.g_na * m**3 * h *
              (v - self.v_na) - self.g_l * (v - self.v_l)) / (100. * self.c_m)

        t_p = self.t_max / (3.3 * np.exp(
            (v + 35.) / 20.) + np.exp(-(v + 35.) / 20.))

        dp = (1. / (1 + np.exp(-(v + 35) / 10.)) - p) / t_p

        a_q = 0.055 * (-27. - v) / (np.exp((-27. - v) / 3.8) - 1.)
        b_q = 0.94 * np.exp((-75. - v) / 17.)
        dq = a_q * (1. - q) - b_q * q

        a_s = 0.000457 * np.exp((-13. - v) / 50.)
        b_s = 0.0065 / (np.exp((-15. - v) / 28.) + 1.)
        ds = a_s * (1. - s) - b_s * s

        a_n = -0.032 * (v - self.v_t -
                        15.) / (np.exp(-(v - self.v_t - 15.) / 5.) - 1)
        b_n = 0.5 * np.exp(-(v - self.v_t - 10.) / 40.)
        dn = a_n * (1. - n) - b_n * n

        a_m = -0.32 * (v - self.v_t -
                       13.) / (np.exp(-(v - self.v_t - 13.) / 4.) - 1)
        b_m = 0.28 * (v - self.v_t - 40.) / (np.exp(
            (v - self.v_t - 40.) / 5.) - 1)
        dm = a_m * (1. - m) - b_m * m

        a_h = 0.128 * np.exp(-(v - self.v_t - 17.) / 18.)
        b_h = 4. / (1. + np.exp(-(v - self.v_t - 40.) / 5.))
        dh = a_h * (1. - h) - b_h * h

        return tuple(self.time_scale * dx
                     for dx in (dv, dp, dq, ds, dn, dm, dh))


class HodgkinHuxleyFFE(Dynamics):
    """
    Two RSA neurons coupled in feedforward with an electrical synapse.
    """

    def __init__(self):
        super().__init__(10, 1, (1, 0, 0, 0, 0, 1, 0, 0, 0, 0))
        self._method = "BDF"

        self.rsa = HodgkinHuxleyRSA()
        self.v_scale = self.rsa.v_scale
        self.time_scale = self.rsa.time_scale
        self.eps = 0.1

    def default_initial_state(self):
        return initial_state.HHFFEInitialState()

    def _dx(self, x, u):
        x_in = x[:5]
        x_out = x[5:]
        delta = self.v_scale * (x_in[0] - x_out[0])

        dx_in = self.rsa._dx(x_in, u)
        dx_out = self.rsa._dx(x_out, self.eps * delta)

        return (*dx_in, *dx_out)


class HodgkinHuxleyFBE(Dynamics):
    """
    Two RSA neurons coupled with an electrical synapse in feedfoward and
    a chemical synapse in feedback.
    """

    def __init__(self):
        super().__init__(11, 1, (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
        self._method = "BDF"

        self.rsa = HodgkinHuxleyRSA()
        self.v_scale = self.rsa.v_scale
        self.time_scale = self.rsa.time_scale

        self.eps_el = 0.1
        self.eps_ch = 0.5

        self.v_syn = 20.
        self.tau_r = 0.5
        self.tau_d = 8.
        self.v0 = -20.

    def default_initial_state(self):
        return initial_state.HHFBEInitialState()

    def _dx(self, x, u):
        x_in = x[:5]
        x_out = x[5:-1]
        r = x[-1]

        v_in = self.v_scale * x_in[0]
        v_out = self.v_scale * x_out[0]

        delta_el = v_in - v_out
        delta_ch = self.v_syn - v_out

        dx_in = self.rsa._dx(x_in, u + r * self.eps_ch * delta_ch)
        dx_out = self.rsa._dx(x_out, self.eps_el * delta_el)

        dr = (1 / self.tau_r - 1 / self.tau_d) * (1. - r) / (
            1. + np.exp(-v_out + self.v0)) - r / self.tau_d

        return (*dx_in, *dx_out, self.time_scale * dr)


class GreenshieldsTraffic(Dynamics):

    def __init__(self, n, v0, dx=None):
        super().__init__(n, 1)
        self.inv_step = self.n if not dx else 1. / dx
        self.v0 = v0

    def flux(self, x):
        return self.v0 * x * (1. - x)

    def _dx(self, x, u):
        q_out = self.flux(x)
        q0_in = self.flux(u.item())

        q_in = np.roll(q_out, 1)
        q_in[0] = q0_in

        dx = self.inv_step * (q_in - q_out)

        return dx


class TwoTank(Dynamics):
    ''' Two tank dynamics with overflow.
        Source: https://apmonitor.com/do/index.php/Main/LevelControl
    '''

    def __init__(self):
        super().__init__(2, 2)
        self._method = "BDF"

        self.c1 = 0.08  # inlet valve coefficient
        self.c2 = 0.04  # outlet valve coefficient

    def default_initial_state(self):
        return initial_state.UniformInitialState(self.n)

    def _dx(self, x, u):
        h1, h2 = x

        pump = u[0]
        valve = u[1]

        dh1 = self.c1 * (1.0 - valve) * pump - self.c2 * np.sqrt(np.abs(h1))
        dh2 = self.c1 * valve * pump + self.c2 * \
            np.sqrt(np.abs(h1)) - self.c2 * np.sqrt(np.abs(h2))

        if (h1 >= 1. and dh1 > 0.) or (h1 <= 1e-10 and dh1 < 0.):
            dh1 = 0.

        if (h2 >= 1. and dh2 > 0.) or (h2 <= 1e-10 and dh2 < 0.):
            dh2 = 0.

        return (dh1, dh2)

class Heat(Dynamics):
    def __init__(self,n,alpha,L,dx=None):
        ''' 1 Dimensional heat equation with spatial temporal inputs
        https://forschung.rwu.de/sites/forschung/files/2024-01/Embedded_Seminar_Report_Scheiter_Lauble_Qureshi.pdf
        '''
        super().__init__(n,2) # 
        self.L = L # rod length [cm]
        self.alpha = alpha # thermal diffusivity  [cm^2/s]
        self.inv_x_step = 1/ (self.L/(n-1)) # 1/delta_x
        self.locations = np.linspace(0, L, n)
        self.locations_orig = np.linspace(0, 100, 50)
        self.sigma = 100/15

    def set_input_mask(self):
        self.first_input = np.random.uniform(0.1*self.L, 0.9*self.L)                        # location of first input
        self.second_input = np.random.uniform(0.1*self.L, 0.9*self.L)                       # location of second input
        self.b_1 = np.exp(-(self.locations -self.first_input)**2 / (2 * self.sigma**2))     # gaussian mask first input
        self.b_1_orig = np.exp(-(self.locations_orig -self.first_input)**2 / (2 * self.sigma**2))     # gaussian mask first input
        self.b_2 = np.exp(-(self.locations -self.second_input)**2 / (2 * self.sigma**2))    # gaussian mask second input
        self.b_2_orig = np.exp(-(self.locations_orig -self.second_input)**2 / (2 * self.sigma**2))    # gaussian mask second input
        self.input_mask = np.column_stack((self.b_1, self.b_2))                             # combine them
        self.input_mask_orig = np.column_stack((self.b_1_orig, self.b_2_orig))                             # combine them

    def _dx(self,x,u):
        a = self.alpha * (self.inv_x_step**2) * (np.roll(x, -1) - 2*x + np.roll(x, 1))
        dt = self.alpha * (self.inv_x_step**2) * (np.roll(x, -1) - 2*x + np.roll(x, 1)) + self.input_mask @ u

        # Neuman boundary conditions, comment if you don't want them to be enforced
        # dt[0] = self.alpha * (self.inv_x_step**2) * 2*(x[1]-x[0])
        # dt[-1] = self.alpha * (self.inv_x_step**2) * 2*(x[-2]-x[-1])

        return dt
    
class Amari(Dynamics):
    def __init__(self,x_lim,dx,theta,kernel_type,kernel_pars):
        '''Amari, S. I. (1977). Dynamics of pattern formation in lateral-inhibition type neural fields. Biological Cybernetics, 27(2), 77-87,'
        'implementation based on: https://github.com/w-wojtak/neural-fields-python?tab=readme-ov-file#1'''

        super().__init__((int(np.round(x_lim/dx))+1,), int(np.round(x_lim/dx)) +1)
        self._method = "SM" # Spectral method, uses FFT-based convolutions

        self.x_lim = x_lim
        self.dx = dx
        self.theta = theta
        self.x = np.arange(0,x_lim+dx,dx)
        self.locations = self.x
        if kernel_type == 0:
            self.w_hat = np.fft.fft(self.kernel_gauss(self.x, *kernel_pars))
        elif kernel_type == 1:
            self.w_hat = np.fft.fft(self.kernel_mex(self.x, *kernel_pars))
        elif kernel_type == 2:
            self.w_hat = np.fft.fft(self.kernel_osc(self.x, *kernel_pars))
        elif kernel_type == 3:
            self.w_hat = np.fft.fft(self.kernel_cos(self.x, *kernel_pars))

    def kernel_mex(self,x, a_ex, s_ex, a_in, s_in, w_in):
        return a_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - a_in * np.exp(-0.5 * x ** 2 / s_in ** 2) - w_in

    def kernel_gauss(self,x, a_ex, s_ex, w_in):
        return a_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - w_in

    def kernel_osc(self,x, a, b, alpha):
        return a * (np.exp(-b*abs(x)) * ((b * np.sin(abs(alpha*x)))+np.cos(alpha*x)))

    def kernel_cos(self, x,A_1,A_2,a_1,a_2):
        return (A_1*np.exp(-a_1* x ** 2 ) - A_2*np.exp(-a_2 * x ** 2)) * np.cos(x/2)

    def simulate(self,u0,inputs,n_samples,time_horizon,init_time):

        dt = (time_horizon-init_time)/inputs.shape[0]
        t = np.arange(init_time,time_horizon,dt)
        history_u = np.zeros([len(t), len(self.x)])
        u_field = u0
        for i in range(0, len(t)):
            f_hat = np.fft.fft(np.heaviside(u_field - self.theta, 1))
            conv = self.dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * self.w_hat)))
            u_field = u_field + dt * (-u_field + conv + inputs[i, :])
            history_u[i, :] = u_field
        
        return history_u, t

class AmariCoupled(Dynamics):
    def __init__(self,x_lim,dx,theta,kernel_types,kernel_pars,diffusion,advection):
        '''Two field Amari, S. I. (1977). Dynamics of pattern formation in lateral-inhibition type neural fields. Biological Cybernetics, 27(2), 77-87,'
        'implementation based on: https://github.com/w-wojtak/neural-fields-python?tab=readme-ov-file#1'''

        super().__init__((int(np.round(x_lim/dx))+1,2), int(np.round(x_lim/dx)) +1)
        self._method = "SM" # Spectral method, uses FFT-based convolutions
        self.x_lim = x_lim
        self.dx = dx
        self.theta = theta
        self.x = np.arange(0,x_lim+dx,dx)
        self.locations = self.x
        self.beta = 2000
        self.tau_v = 5 # change 
        self.tau_u = self.tau_v/5 # change 
        self.diffusion = bool(diffusion)
        self.advection = bool(advection)
        self.w_hat = []

        # kernels for u and v
        for (kernel_type,kernel_par) in zip(kernel_types,kernel_pars):
            if kernel_type == 0:
                self.w_hat.append(np.fft.fft(self.kernel_gauss(self.x, *kernel_par)))
            elif kernel_type == 1:
                self.w_hat.append(np.fft.fft(self.kernel_mex(self.x, *kernel_par)))
            elif kernel_type == 2:
                self.w_hat.append(np.fft.fft(self.kernel_osc(self.x, *kernel_par)))
            elif kernel_type == 3:
                self.w_hat.append(np.fft.fft(self.kernel_cos(self.x, *kernel_par)))

    def kernel_mex(self,x, a_ex, s_ex, a_in, s_in, w_in):
        return a_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - a_in * np.exp(-0.5 * x ** 2 / s_in ** 2) - w_in

    def kernel_gauss(self,x, a_ex, s_ex, w_in):
        return a_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - w_in

    def kernel_osc(self,x, a, b, alpha):
        return a * (np.exp(-b*abs(x)) * ((b * np.sin(abs(alpha*x)))+np.cos(alpha*x)))
    
    def kernel_cos(self, x,A_1,A_2,a_1,a_2):
        return (A_1*np.exp(-a_1* x ** 2 ) - A_2*np.exp(-a_2 * x ** 2)) * np.cos(x/2)

    def sigmoid(self,x, beta,theta):
        return 1 / (1 + np.exp(-beta * (x - theta)))

    def simulate(self,x0,inputs,n_samples,time_horizon,init_time):

        dt = (time_horizon-init_time)/inputs.shape[0]
        t = np.arange(init_time,time_horizon,dt)
        history_u = np.zeros([len(t), len(self.x)])
        history_v = np.zeros([len(t), len(self.x)])

        # IC
        u_field = x0[:,0]
        v_field = x0[:,1]

        
        k = np.fft.fftfreq(len(self.x), self.dx) * 2 * np.pi  # Wave numbers in Fourier space

        u_field_full = []
        v_field_full = []
        for i in range(0, len(t)):
            f_hat = self.sigmoid(u_field, self.beta, self.theta)
            conv_u = self.dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * self.w_hat[0])))
            conv_v = self.dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * self.w_hat[1])))

            # Compute the second derivative of v using the Fourier domain representation
            v_hat = np.fft.fft(v_field)  # Fourier transform of v_field
            u_hat = np.fft.fft(u_field)  # Fourier transform of v_field
            if self.advection:
                advection_term = np.real(np.fft.ifft(1j *k* v_hat))  # Multiply by k^2 for the second derivative in Fourier space
            else:
                advection_term = 0

            if self.diffusion:
                diffusion_term = np.real(np.fft.ifft(-k**2 * v_hat))  # Multiply by k^2 for the second derivative in Fourier space
            else:
                diffusion_term = 0
          
            

            u_field += dt /self.tau_u * (-u_field + conv_u + v_field + inputs[i, :])
            u_field_full.append(u_field)
            v_field += dt / self.tau_v * (-v_field - conv_v + u_field  + 10**-3 * diffusion_term + 10**-2 *v_field*advection_term)
            v_field_full.append(v_field)
            history_u[i, :] = u_field
            history_v[i, :] = v_field
        
        return np.stack((history_u,history_v),axis=-1), t


_dynamics_names = {
    "LinearSys": LinearSys,
    "VanDerPol": VanDerPol,
    "FitzHughNagumo": FitzHughNagumo,
    "Pendulum": Pendulum,
    "HodgkinHuxleyFS": HodgkinHuxleyFS,
    "HodgkinHuxleyRSA": HodgkinHuxleyRSA,
    "HodgkinHuxleyIB": HodgkinHuxleyIB,
    "HodgkinHuxleyFFE": HodgkinHuxleyFFE,
    "HodgkinHuxleyFBE": HodgkinHuxleyFBE,
    "GreenshieldsTraffic": GreenshieldsTraffic,
    "TwoTank": TwoTank,
    "Heat": Heat,
    "Amari":Amari,
    "AmariCoupled":AmariCoupled,
}


def get_dynamics(name, args):
    return _dynamics_names[name](**args)
